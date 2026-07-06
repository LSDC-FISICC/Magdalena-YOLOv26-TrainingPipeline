#include "InferenceNode_v2.hpp"



InferenceNode_v2::InferenceNode_v2() : Node("Yolo_InferenceNode_v2"){

    //Engine_path definition
    this->declare_parameter<std::string>("engine_path", "../../src/best.engine");
    std::string enginePath = this->get_parameter("engine_path").as_string();


    // Initialisation of the TensoRT model
    try {
        m_model = std::make_unique<model>(enginePath, m_expectedBatchSize);
        RCLCPP_INFO(this->get_logger(), "Model loaded. Max batch size set to %d", m_expectedBatchSize);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Failed to load model: %s", e.what());
        throw e;
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


//Function to infer the received batch from camera/preprocessed_frames
void InferenceNode_v2::imageCallback(const sensor_msgs::msg::Image::SharedPtr msg){
    // Extract timestamp and camera_id
    rclcpp::Time msg_time = msg->header.stamp;
    std::string camera_name = msg->header.frame_id; 

    //split camera_name to extract each camera_id
    std::vector<std::string> batch_camera_name;
    std::stringstream ss(camera_name);
    std::string item;
    while(std::getline(ss, item, ',')){
        if(!item.empty()){
            batch_camera_name.push_back(item);
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

        int i = 0;
        // Global status with every keys/values
        diagnostic_msgs::msg::DiagnosticArray batch_status_msg;
        batch_status_msg.header.stamp = msg_time;
        batch_status_msg.header.frame_id = "plant_detection_from_camera";

        
        
        for(i; i<results.size();  ++i){


            std::string camera_name = batch_camera_name[i];
            diagnostic_msgs::msg::KeyValue state_msg;
            state_msg.key = "plant_detected";

            //DiagnosticStatus for the detection for each camera_id
            diagnostic_msgs::msg::DiagnosticStatus array_container;
            array_container.name = camera_name ; 
            array_container.level = diagnostic_msgs::msg::DiagnosticStatus::OK;

            if(results[i].predictedClass == 1){
                classe_name = "plant";
                state_msg.value = "true";
            }else if(results[i].predictedClass == 0){
                classe_name = "no_plant";
                state_msg.value = "false";
            }

            array_container.values.push_back(state_msg);
            batch_status_msg.status.push_back(array_container);
            

            RCLCPP_INFO(this->get_logger(), "Topic: %s -> Pred: %d , %s,  (Conf: %.2f)", 
                    camera_name.c_str(), results[i].predictedClass, classe_name.c_str(), results[i].confidence);
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

