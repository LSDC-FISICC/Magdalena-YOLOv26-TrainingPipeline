#include "InferenceNode.hpp"

static const std::vector<std::string> CAMERAS_TOPIS{
    "/camera/camera/color/image_raw",
};

InferenceNode::InferenceNode() : Node("Yolo_InferenceNode"), m_cameraTopics(CAMERAS_TOPIS) {


    this->declare_parameter<std::string>("engine_path", "../../src/best.engine");
    std::string enginePath = this->get_parameter("engine_path").as_string();

    // La taille max du batch correspond au nombre de caméras configurées (max 5 ici)
    int maxBatchSize = std::min(static_cast<int>(m_cameraTopics.size()), 5);
    if (maxBatchSize == 0) {
        RCLCPP_ERROR(this->get_logger(), "No camera topics configured in CAMERA_TOPICS!");
        return;
    }

    // 2. Initialisation du modèle TensorRT
    try {
        m_model = std::make_unique<model>(enginePath, maxBatchSize);
        RCLCPP_INFO(this->get_logger(), "Model loaded. Max batch size set to %d", maxBatchSize);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Failed to load model: %s", e.what());
        throw e;
    }

    // 3. Création DYNAMIQUE des abonnements (Subscriptions)
    for (const auto& topic : m_cameraTopics) {
        // On utilise une fonction lambda pour passer le nom du topic au callback
        auto callback = [this, topic](const sensor_msgs::msg::Image::SharedPtr msg) {
            this->imageCallback(msg, topic);
        };

        RCLCPP_INFO(this->get_logger(), "Subscribing to: %s", topic.c_str());
        m_subscriptions.push_back(
            this->create_subscription<sensor_msgs::msg::Image>(topic, 10, callback)
        );
    }

    // 4. Déclenchement du timer pour l'inférence (ex: 30Hz -> toutes les 33ms ou 10Hz -> 100ms)
    // Ajuste la fréquence selon la puissance de ton GPU et le besoin de ton application
    m_timer = this->create_wall_timer(
        std::chrono::milliseconds(67), // 15 Hz
        std::bind(&InferenceNode::timerCallback, this)
    );


}

void InferenceNode::imageCallback(const sensor_msgs::msg::Image::SharedPtr msg, const std::string& topic_name){
    std::lock_guard<std::mutex> lock(m_bufferMutex);
    m_imageBuffer[topic_name] = msg ;

}

void InferenceNode::timerCallback(){
    std::vector<cv::Mat> cvImages ;
    std::vector<std::string> activeTopics;

    {
        std::lock_guard<std::mutex> lock(m_bufferMutex) ; 

        for(const auto& topic : m_cameraTopics){
            if(m_imageBuffer.count(topic) && m_imageBuffer[topic]!= nullptr){

                try{
                    cv::Mat cvImage = cv_bridge::toCvShare(m_imageBuffer[topic], "bgr8")->image;
                    cvImages.push_back(cvImage);
                    activeTopics.push_back(topic);


                }catch(const cv_bridge::Exception & e){
                    RCLCPP_ERROR(this->get_logger(), "cv_bridge conversion failed for %s: %s", topic.c_str(), e.what());

                }

                m_imageBuffer[topic] = nullptr;
            }

        }

        if(cvImages.empty()){
            return;
        }
        std::string classe_name;
        try{
            std::vector<InferenceResult> results = m_model->analyzeBatch(cvImages);

            int i = 0;

            for(i; i<results.size();  ++i){
                if(results[i].predictedClass == 1){
                    classe_name = "plant";
                }else if(results[i].predictedClass == 0){
                    classe_name = "no_plant";
                }
                RCLCPP_INFO(this->get_logger(), "Topic: %s -> Pred: %d , %s,  (Conf: %.2f)", 
                        activeTopics[i].c_str(), results[i].predictedClass, classe_name.c_str(), results[i].confidence);
            }

        }catch(const std::exception& e){
            RCLCPP_ERROR(this->get_logger(), "Error during TensorRT inference: %s", e.what());
        }
    }
}

