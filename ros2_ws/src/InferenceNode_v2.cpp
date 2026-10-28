#include "InferenceNode_v2.hpp"



InferenceNode_v2::InferenceNode_v2() : Node("Yolo_InferenceNode_v2"){

    //Engine_path definition
    this->declare_parameter<std::string>("engine_path", "../../src/best.engine");
    std::string enginePath = this->get_parameter("engine_path").as_string();
    // Must not exceed the max shape the engine was built with (convert2engine4jetson_v2.py: 5)
    m_expectedBatchSize = this->declare_parameter<int>("max_batch_size", 5);


    // Initialisation of the TensoRT model
    try {
        m_model = std::make_unique<model>(enginePath, m_expectedBatchSize);
        RCLCPP_INFO(this->get_logger(), "Model loaded. Max batch size set to %d", m_expectedBatchSize);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Failed to load model: %s", e.what());
        throw e;
    }

    // A threshold below 0.5 trades a little saving for less cane left without vinasse
    m_plantThreshold = this->declare_parameter<double>("plant_threshold", 0.5);
    RCLCPP_INFO(this->get_logger(), "plant_threshold %.2f", m_plantThreshold);

    // Optional fallback engine for blurred passes (night, long exposure); see InferenceNode_v2.hpp
    std::string fallbackPath = this->declare_parameter<std::string>("fallback_engine_path", "");
    // 0.561: Otsu threshold of blurRatio() on unlabelled day, blurred and night 320x94 crops
    m_blurThreshold = this->declare_parameter<double>("blur_threshold", 0.561);
    m_blurWindow = static_cast<size_t>(std::max<int64_t>(1, this->declare_parameter<int64_t>("blur_window", 30)));
    if (!fallbackPath.empty()) {
        m_fallback = std::make_unique<model>(fallbackPath, m_expectedBatchSize);
        RCLCPP_INFO(this->get_logger(), "Fallback model %s used while the median blur ratio of the last %zu frames < %.3f",
                    fallbackPath.c_str(), m_blurWindow, m_blurThreshold);
    }

    // Loading the subscription
    m_sharedImg = this->create_subscription<sensor_msgs::msg::Image>(
    "/camera/preprocessed_frames", 
    10, 
    std::bind(&InferenceNode_v2::imageCallback, this, std::placeholders::_1)
    );

    //Publisher
    m_batchPlantPub = this->create_publisher<diagnostic_msgs::msg::DiagnosticArray>("/detection/Trigger", 10);

}


// Blur along the direction of travel (vertical in the crop) removes vertical gradients only:
// mean |dI/dy| / mean |dI/dx| of the grey crop, independent of brightness and contrast.
// Same statistic as the offline analysis (Otsu threshold 0.561, about 0.8 on sharp day frames).
float InferenceNode_v2::blurRatio(const cv::Mat& img){
    cv::Mat f, gray;
    img.convertTo(f, CV_32F);
    cv::cvtColor(f, gray, cv::COLOR_BGR2GRAY);
    const double dy = cv::mean(cv::abs(gray.rowRange(1, gray.rows) - gray.rowRange(0, gray.rows - 1)))[0];
    const double dx = cv::mean(cv::abs(gray.colRange(1, gray.cols) - gray.colRange(0, gray.cols - 1)))[0];
    return static_cast<float>(dy / std::max(dx, 1e-9));
}


//Function to infer the received batch from camera/preprocessed_frames
void InferenceNode_v2::imageCallback(const sensor_msgs::msg::Image::SharedPtr msg){
    const int64_t t_received = this->get_clock()->now().nanoseconds();
    // Extract timestamp and camera_id
    rclcpp::Time msg_time = msg->header.stamp;
    std::string camera_name = msg->header.frame_id; 

    //split camera_name to extract each camera_id ("topic@capture_stamp_ns" since cameraNode
    //propagates capture stamps; a plain "topic" is still accepted)
    std::vector<std::string> batch_camera_name;
    std::vector<std::string> batch_capture_ns;
    std::stringstream ss(camera_name);
    std::string item;
    while(std::getline(ss, item, ',')){
        if(!item.empty()){
            size_t at = item.find('@');
            batch_camera_name.push_back(item.substr(0, at));
            batch_capture_ns.push_back(at == std::string::npos ? "" : item.substr(at + 1));
        }
    }
    if(batch_camera_name.empty()){
        RCLCPP_ERROR(this->get_logger(), "No Id found in camera_id");
        return;

    }

    // Convert the received concatenated images into cv::Mat
    cv::Mat img;
    try {
        // Decoding with the received encoding message
        std::string encoding = msg->encoding;
        img = cv_bridge::toCvCopy(msg, encoding)->image;
    } catch (const cv_bridge::Exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Erreur conversion cv_bridge à la réception: %s", e.what());
        return;
    }

    //Generate a std::vector with the images of each camera_id by spliting the concatenated cv::mat
    std::vector<cv::Mat> batch_images;
    int img_h = 96;
    int img_w = 320;

    for(size_t i = 0; i< batch_camera_name.size(); ++i){
        int offset_h = i * img_h;

        if (offset_h + img_h <= img.rows) {
            cv::Rect crop_zone(0, offset_h, img_w, img_h);
            batch_images.push_back(img(crop_zone));
        } else {
            RCLCPP_ERROR(this->get_logger(), "Taille de l'image reçue incompatible avec le nombre d'IDs détectés.");
            return;
        }
    }

    //Do the inference of the batch
    std::string classe_name;
    try{
        std::vector<InferenceResult> results = m_model->analyzeBatch_preprocessed(batch_images);
        std::vector<InferenceResult> fallback;
        if (m_fallback) {
            fallback = m_fallback->analyzeBatch_preprocessed(batch_images);
        }
        const int64_t t_infer_done = this->get_clock()->now().nanoseconds();

        int i = 0;
        // Global status with every keys/values
        diagnostic_msgs::msg::DiagnosticArray batch_status_msg;
        batch_status_msg.header.stamp = msg_time;
        batch_status_msg.header.frame_id = "plant_detection_from_camera";

        
        
        for(i; i<results.size();  ++i){


            std::string camera_name = batch_camera_name[i];

            // Blur of this camera's pass: median ratio over its last m_blurWindow frames
            const float ratio = blurRatio(batch_images[i]);
            auto& hist = m_blurHistory[camera_name];
            hist.push_back(ratio);
            while (hist.size() > m_blurWindow) hist.pop_front();
            std::vector<float> sorted(hist.begin(), hist.end());
            std::nth_element(sorted.begin(), sorted.begin() + sorted.size() / 2, sorted.end());
            const float blurMedian = sorted[sorted.size() / 2];
            const bool useFallback = m_fallback && blurMedian < m_blurThreshold;
            if (m_fallback && useFallback != m_onFallback[camera_name]) {
                RCLCPP_INFO(this->get_logger(), "%s: %s model (median blur ratio %.3f)", camera_name.c_str(),
                            useFallback ? "fallback" : "main", blurMedian);
                m_onFallback[camera_name] = useFallback;
            }
            const InferenceResult& chosen = useFallback ? fallback[i] : results[i];
            // two-class softmax: p(plant) from the winning class and its probability
            auto pPlant = [](const InferenceResult& r) { return r.predictedClass == 1 ? r.confidence : 1.0f - r.confidence; };
            const bool plant = pPlant(chosen) >= m_plantThreshold;

            diagnostic_msgs::msg::KeyValue state_msg;
            state_msg.key = "plant_detected";

            //DiagnosticStatus for the detection for each camera_id
            diagnostic_msgs::msg::DiagnosticStatus array_container;
            array_container.name = camera_name ; 
            array_container.level = diagnostic_msgs::msg::DiagnosticStatus::OK;

            if(plant){
                classe_name = "plant";
                state_msg.value = "true";
            }else{
                classe_name = "no_plant";
                state_msg.value = "false";
            }

            array_container.values.push_back(state_msg);

            // Timing and confidence for the latency decomposition (experiments/e7_e2e_latency.py)
            auto kv = [](const std::string& k, const std::string& v) {
                diagnostic_msgs::msg::KeyValue m; m.key = k; m.value = v; return m;
            };
            array_container.values.push_back(kv("confidence", std::to_string(chosen.confidence)));
            array_container.values.push_back(kv("p_plant", std::to_string(pPlant(chosen))));
            array_container.values.push_back(kv("blur_ratio", std::to_string(ratio)));
            array_container.values.push_back(kv("blur_median", std::to_string(blurMedian)));
            if (m_fallback) {
                array_container.values.push_back(kv("model", useFallback ? "fallback" : "main"));
                array_container.values.push_back(kv("plant_main", pPlant(results[i]) >= m_plantThreshold ? "true" : "false"));
                array_container.values.push_back(kv("plant_fallback", pPlant(fallback[i]) >= m_plantThreshold ? "true" : "false"));
            }
            array_container.values.push_back(kv("t_capture_ns", batch_capture_ns[i]));
            array_container.values.push_back(kv("t_preprocessed_ns", std::to_string(msg_time.nanoseconds())));
            array_container.values.push_back(kv("t_received_ns", std::to_string(t_received)));
            array_container.values.push_back(kv("t_infer_done_ns", std::to_string(t_infer_done)));
            batch_status_msg.status.push_back(array_container);
            

            RCLCPP_DEBUG(this->get_logger(), "Topic: %s -> Pred: %d , %s,  (Conf: %.2f)", 
                    camera_name.c_str(), plant ? 1 : 0, classe_name.c_str(), chosen.confidence);
            }

        
        m_batchPlantPub->publish(batch_status_msg);

        }catch(const std::exception& e){
            RCLCPP_ERROR(this->get_logger(), "Error during TensorRT inference: %s", e.what());
        }


    // 3. Stocker l'image dans le collecteur sous sa clé temporelle
    //m_batchCollector[msg_time].push_back(std::make_pair(camera_name, img));

    // 4. Si nous avons reçu toutes les images attendues pour ce timestamp, on lance TensorRT !
    //if (m_batchCollector[msg_time].size() == m_expectedBatchSize) {
        
        // Préparation des vecteurs plats pour ton modèle
        //std::vector<cv::Mat> batch_images;
        //std::vector<std::string> batch_camera_names;

        //for (const auto& pair : m_batchCollector[msg_time]) {
            //batch_camera_names.push_back(pair.first); // Le nom de la caméra
            //batch_images.push_back(pair.second);       // La matrice OpenCV
        //}

        //RCLCPP_INFO(this->get_logger(), "Batch complet reçu pour le timestamp %f. Lancement de l'inférence...", msg_time.seconds());

        //std::string classe_name;
        //try{
            //std::vector<InferenceResult> results = m_model->analyzeBatch_preprocessed(batch_images);

            //int i = 0;

            //diagnostic_msgs::msg::DiagnosticArray batch_status_msg;
            //batch_status_msg.header.stamp = msg_time; // On garde le timestamp d'origine du batch !
            //batch_status_msg.header.frame_id = "camera_array";

            // Un seul statut global qui va contenir notre tableau de clés/valeurs
            //diagnostic_msgs::msg::DiagnosticStatus array_container;
            //array_container.name = "yolo_batch_detections";



            //for(i; i<results.size();  ++i){
                //std::string camera_name = batch_camera_names[i];
                //diagnostic_msgs::msg::KeyValue state_msg;
                //state_msg.key = camera_name;

                //if(results[i].predictedClass == 1){
                    //classe_name = "plant";
                    //state_msg.value = "true";
                //}else if(results[i].predictedClass == 0){
                    //classe_name = "no_plant";
                    //state_msg.value = "false";
                //}

                //array_container.values.push_back(state_msg);

                //RCLCPP_INFO(this->get_logger(), "Topic: %s -> Pred: %d , %s,  (Conf: %.2f)", 
                        //camera_name.c_str(), results[i].predictedClass, classe_name.c_str(), results[i].confidence);
            //}

            //batch_status_msg.status.push_back(array_container);
            //m_batchPlantPub->publish(batch_status_msg);

        //}catch(const std::exception& e){
            //RCLCPP_ERROR(this->get_logger(), "Error during TensorRT inference: %s", e.what());
        //}
        

        //m_batchCollector.erase(msg_time);
    //}

    //auto now = this->get_clock()->now();
    //for (auto it = m_batchCollector.begin(); it != m_batchCollector.end(); ) {
        //if ((now - it->first).seconds() > 1.0) {
            //RCLCPP_WARN(this->get_logger(), "Batch temporel incomplet expiré abandonné.");
            //it = m_batchCollector.erase(it);
        //} else {
            //++it;
        //}
    //}

}

