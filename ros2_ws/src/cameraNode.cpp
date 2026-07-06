#include "cameraNode.hpp"

static const std::vector<std::string> CAMERAS_TOPICS {
    "/camera/camera/color/image_raw",
};

cameraNode::cameraNode() : Node("camera_process_node"), m_cameraTopics(CAMERAS_TOPICS){

    // Build the subscription
    for (const auto& topic : m_cameraTopics) {
        auto callback = [this, topic](const sensor_msgs::msg::Image::SharedPtr msg) {
            this->imageCallback(msg, topic);
        };

        RCLCPP_INFO(this->get_logger(), "Subscribing to: %s", topic.c_str());
        m_subscriptions.push_back(
            this->create_subscription<sensor_msgs::msg::Image>(topic, 10, callback)
        );
    }
    // creation of the publishers
    
    m_preprocessedImgPub = this->create_publisher<sensor_msgs::msg::Image>("/camera/preprocessed_frames", 10);

    // Turn on the timer for the preprocess
    m_timer = this->create_wall_timer(
        std::chrono::milliseconds(67), // 15 Hz same as the cameras
        std::bind(&cameraNode::timerCallback, this)
    );



}

// receive the images from the cameras and save them in the buffer(m_imageBuffer)
void cameraNode::imageCallback(const sensor_msgs::msg::Image::SharedPtr msg, const std::string& topic_name){
    std::lock_guard<std::mutex> lock(m_bufferMutex);
    m_imageBuffer[topic_name] = msg ;
}


void cameraNode::timerCallback(){   
    //vector of preprocessed images
    std::vector<cv::Mat> cvImages;
    //Lists of Topics
    std::vector<std::string> activeTopics;

    {
        std::lock_guard<std::mutex> lock(m_bufferMutex) ; 

        for(const auto& topic : m_cameraTopics){
            if(m_imageBuffer.count(topic) && m_imageBuffer[topic]!= nullptr){
                // adapt the images for the model dimensions before pushing them to cvImages
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

    cv::Mat batchImages ;
    //concatenate all the images of the batch
    cv::vconcat (cvImages, batchImages) ; 

    std::string encoding = (batchImages.type() == CV_32FC3) ? "32FC3" : "brg8" ;

    auto msg = cv_bridge::CvImage(std_msgs::msg::Header(), encoding, batchImages).toImageMsg();

    //List of Camera_Id
    std::string combined_id = "";

    for(size_t i = 0; i<activeTopics.size(); ++i){
        combined_id += activeTopics[i];
        if(i<activeTopics.size()-1){
            combined_id+=",";
        }
    }

    msg->header.stamp = timestamp;
    msg->header.frame_id = combined_id;
    m_preprocessedImgPub->publish(*msg);
    RCLCPP_INFO(this->get_logger(), "%zu images pré-traitées envoyées.", cvImages.size());

    //send the images to the publisher with their encoding, their camera_id and their timestamp.
    //for(size_t i = 0; i<cvImages.size(); ++i){
      //  cv::Mat img_to_publish = cvImages[i]; 

        //std::string encoding = (img_to_publish.type() == CV_32FC3) ? "32FC3" : "bgr8";

        //auto msg = cv_bridge::CvImage(std_msgs::msg::Header(), encoding, img_to_publish).toImageMsg();

        //msg->header.stamp = timestamp;
        //msg->header.frame_id = activeTopics[i];

        //m_preprocessedImgPub->publish(*msg);


    //}
    //RCLCPP_INFO(this->get_logger(), "%zu images pré-traitées envoyées.", cvImages.size());

}


int main(int argc, char * argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<cameraNode>());
    rclcpp::shutdown();
    return 0;
}