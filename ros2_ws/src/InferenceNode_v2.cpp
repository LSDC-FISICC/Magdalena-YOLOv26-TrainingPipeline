#include "InferenceNode_v2.hpp"



InferenceNode_v2::InferenceNode_v2() : Node("Yolo_InferenceNode_v2"){


    this->declare_parameter<std::string>("engine_path", "../../src/best.engine");
    std::string enginePath = this->get_parameter("engine_path").as_string();


    // 2. Initialisation du modèle TensorRT
    try {
        m_model = std::make_unique<model>(enginePath, m_expectedBatchSize);
        RCLCPP_INFO(this->get_logger(), "Model loaded. Max batch size set to %d", m_expectedBatchSize);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Failed to load model: %s", e.what());
        throw e;
    }

    // 3. Création DYNAMIQUE des abonnements (Subscriptions)
    m_sharedImg = this->create_subscription<sensor_msgs::msg::Image>(
    "/camera/preprocessed_frames", 
    10, 
    std::bind(&InferenceNode_v2::imageCallback, this, std::placeholders::_1)
    );

    //publisher
    m_batchPlantPub = this->create_publisher<diagnostic_msgs::msg::DiagnosticArray>("/detection/Trigger", 10);

}

void InferenceNode_v2::imageCallback(const sensor_msgs::msg::Image::SharedPtr msg){
    // 1. Extraire les métadonnées cruciales
    rclcpp::Time msg_time = msg->header.stamp;
    std::string camera_name = msg->header.frame_id; // Contient par exemple "/camera_gauche/image_raw"

    // 2. Convertir le message ROS en cv::Mat
    cv::Mat img;
    try {
        // On vérifie l'encodage selon ce que tu as envoyé (32FC3 ou bgr8)
        std::string encoding = msg->encoding;
        img = cv_bridge::toCvCopy(msg, encoding)->image;
    } catch (const cv_bridge::Exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Erreur conversion cv_bridge à la réception: %s", e.what());
        return;
    }

    // 3. Stocker l'image dans le collecteur sous sa clé temporelle
    m_batchCollector[msg_time].push_back(std::make_pair(camera_name, img));

    // 4. Si nous avons reçu toutes les images attendues pour ce timestamp, on lance TensorRT !
    if (m_batchCollector[msg_time].size() == m_expectedBatchSize) {
        
        // Préparation des vecteurs plats pour ton modèle
        std::vector<cv::Mat> batch_images;
        std::vector<std::string> batch_camera_names;

        for (const auto& pair : m_batchCollector[msg_time]) {
            batch_camera_names.push_back(pair.first); // Le nom de la caméra
            batch_images.push_back(pair.second);       // La matrice OpenCV
        }

        //RCLCPP_INFO(this->get_logger(), "Batch complet reçu pour le timestamp %f. Lancement de l'inférence...", msg_time.seconds());

        std::string classe_name;
        try{
            std::vector<InferenceResult> results = m_model->analyzeBatch_preprocessed(batch_images);

            int i = 0;

            diagnostic_msgs::msg::DiagnosticArray batch_status_msg;
            batch_status_msg.header.stamp = msg_time; // On garde le timestamp d'origine du batch !
            batch_status_msg.header.frame_id = "camera_array";

            // Un seul statut global qui va contenir notre tableau de clés/valeurs
            diagnostic_msgs::msg::DiagnosticStatus array_container;
            array_container.name = "yolo_batch_detections";



            for(i; i<results.size();  ++i){
                std::string camera_name = batch_camera_names[i];
                diagnostic_msgs::msg::KeyValue state_msg;
                state_msg.key = camera_name;

                if(results[i].predictedClass == 1){
                    classe_name = "plant";
                    state_msg.value = "true";
                }else if(results[i].predictedClass == 0){
                    classe_name = "no_plant";
                    state_msg.value = "false";
                }

                array_container.values.push_back(state_msg);

                RCLCPP_INFO(this->get_logger(), "Topic: %s -> Pred: %d , %s,  (Conf: %.2f)", 
                        camera_name.c_str(), results[i].predictedClass, classe_name.c_str(), results[i].confidence);
            }

            batch_status_msg.status.push_back(array_container);
            m_batchPlantPub->publish(batch_status_msg);

        }catch(const std::exception& e){
            RCLCPP_ERROR(this->get_logger(), "Error during TensorRT inference: %s", e.what());
        }
        

        m_batchCollector.erase(msg_time);
    }

    auto now = this->get_clock()->now();
    for (auto it = m_batchCollector.begin(); it != m_batchCollector.end(); ) {
        if ((now - it->first).seconds() > 1.0) {
            RCLCPP_WARN(this->get_logger(), "Batch temporel incomplet expiré abandonné.");
            it = m_batchCollector.erase(it);
        } else {
            ++it;
        }
    }

}

