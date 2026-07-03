#include "cameraNode.hpp"

static const std::vector<std::string> CAMERAS_TOPICS {
    "/camera/camera/color/image_raw",
};

cameraNode::cameraNode() : Node("camera_process_node"), m_cameraTopics(CAMERAS_TOPICS){

    // Création DYNAMIQUE des abonnements (Subscriptions)
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

    m_preprocessedImgPub = this->create_publisher<sensor_msgs::msg::Image>("/camera/preprocessed_frames", 10);

    // Déclenchement du timer pour l'inférence (ex: 30Hz -> toutes les 33ms ou 10Hz -> 100ms)
    // Ajuste la fréquence selon la puissance de ton GPU et le besoin de ton application
    m_timer = this->create_wall_timer(
        std::chrono::milliseconds(67), // 15 Hz
        std::bind(&cameraNode::timerCallback, this)
    );



}


void cameraNode::imageCallback(const sensor_msgs::msg::Image::SharedPtr msg, const std::string& topic_name){
    std::lock_guard<std::mutex> lock(m_bufferMutex);
    m_imageBuffer[topic_name] = msg ;

}

void cameraNode::timerCallback(){   

    std::vector<cv::Mat> cvImages;
    std::vector<std::string> activeTopics;

    {
        std::lock_guard<std::mutex> lock(m_bufferMutex) ; 

        for(const auto& topic : m_cameraTopics){
            if(m_imageBuffer.count(topic) && m_imageBuffer[topic]!= nullptr){

                try{
                    cv::Mat cvImage = cv_bridge::toCvShare(m_imageBuffer[topic], "bgr8")->image;
                    int height = cvImage.rows;
                    int weights = cvImage.cols;
                    int x_start = weights/2 - 160;
                    int y_start = height/2 -48;
                    cv::Rect rectangle(x_start, y_start, 320, 96);

                    cv::Mat img_rect = cvImage(rectangle);
                    cv::Mat resized;
                    cv::resize(img_rect, resized, cv::Size(320, 96));

                    cv::Mat imgFloat; 
                    resized.convertTo(imgFloat, CV_32FC3, 1.0f/ 255.0f);

                    cvImages.push_back(imgFloat);
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

    }
    auto timestamp = this->get_clock()->now();

    for(size_t i = 0; i<cvImages.size(); ++i){
        cv::Mat img_to_publish = cvImages[i]; 

        std::string encoding = (img_to_publish.type() == CV_32FC3) ? "32FC3" : "bgr8";

        auto msg = cv_bridge::CvImage(std_msgs::msg::Header(), encoding, img_to_publish).toImageMsg();

        msg->header.stamp = timestamp;
        msg->header.frame_id = activeTopics[i];

        m_preprocessedImgPub->publish(*msg);


    }
    RCLCPP_INFO(this->get_logger(), "%zu images pré-traitées envoyées.", cvImages.size());

}


int main(int argc, char * argv[]) {
    // Initialise ROS 2 pour ce processus
    rclcpp::init(argc, argv);
    
    // Lance la boucle infinie (spin) spécifique au nœud de préprocessing
    rclcpp::spin(std::make_shared<cameraNode>());
    
    // Nettoie proprement à la fermeture (Ctrl+C)
    rclcpp::shutdown();
    return 0;
}