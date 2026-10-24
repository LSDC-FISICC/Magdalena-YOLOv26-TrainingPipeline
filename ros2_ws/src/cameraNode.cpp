#include "cameraNode.hpp"

static const std::vector<std::string> CAMERAS_TOPICS {
    "/camera/camera/color/image_raw",
};

cameraNode::cameraNode() : Node("camera_process_node"){

    // The batch order sent to the inference node follows this list (camera i -> furrow i).
    // Magdalena-Camera publishes /camera_<ip>/color/image_raw, one entry per D555.
    m_cameraTopics = this->declare_parameter<std::vector<std::string>>("camera_topics", CAMERAS_TOPICS);
    // Crop must match how the training frames were cut before Roboflow stretched them to 320x96
    m_roiX = this->declare_parameter<int>("roi_x", -1);
    m_roiY = this->declare_parameter<int>("roi_y", -1);
    m_roiW = this->declare_parameter<int>("roi_w", -1);
    m_roiH = this->declare_parameter<int>("roi_h", -1);
    double rate_hz = this->declare_parameter<double>("rate_hz", 15.0);

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
        std::chrono::duration<double>(1.0 / rate_hz), // 15 Hz same as the cameras
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
    //Capture stamp of each frame, propagated for the camera-to-valve latency measurement
    std::vector<int64_t> captureStamps;

    {
        std::lock_guard<std::mutex> lock(m_bufferMutex) ; 

        for(const auto& topic : m_cameraTopics){
            if(m_imageBuffer.count(topic) && m_imageBuffer[topic]!= nullptr){
                // adapt the images for the model dimensions before pushing them to cvImages
                try{
                    cv::Mat cvImage = cv_bridge::toCvShare(m_imageBuffer[topic], "bgr8")->image;
                    int height = cvImage.rows;
                    int weights = cvImage.cols;
                    cv::Rect rectangle(weights/2 - 160, height/2 - 48, 320, 96);
                    if (m_roiW > 0 && m_roiH > 0) {
                        rectangle = cv::Rect(m_roiX < 0 ? (weights - m_roiW) / 2 : m_roiX,
                                             m_roiY < 0 ? (height - m_roiH) / 2 : m_roiY,
                                             m_roiW, m_roiH);
                    }
                    rectangle &= cv::Rect(0, 0, weights, height);

                    cv::Mat img_rect = cvImage(rectangle);
                    cv::Mat resized;
                    cv::resize(img_rect, resized, cv::Size(320, 96));

                    cv::Mat imgFloat; 
                    resized.convertTo(imgFloat, CV_32FC3, 1.0f/ 255.0f);

                    cvImages.push_back(imgFloat);
                    activeTopics.push_back(topic);
                    captureStamps.push_back(rclcpp::Time(m_imageBuffer[topic]->header.stamp).nanoseconds());


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
    cv::Mat batchImages ;
    //concatenate all the images of the batch
    cv::vconcat (cvImages, batchImages) ; 

    std::string encoding = (batchImages.type() == CV_32FC3) ? "32FC3" : "bgr8" ;

    auto msg = cv_bridge::CvImage(std_msgs::msg::Header(), encoding, batchImages).toImageMsg();

    //List of Camera_Id with the capture stamp of each frame: "topic@stamp_ns,topic@stamp_ns"
    std::string combined_id = "";

    for(size_t i = 0; i<activeTopics.size(); ++i){
        combined_id += activeTopics[i] + "@" + std::to_string(captureStamps[i]);
        if(i<activeTopics.size()-1){
            combined_id+=",";
        }
    }

    // stamp = end of preprocessing; capture stamps travel in frame_id
    msg->header.stamp = this->get_clock()->now();
    msg->header.frame_id = combined_id;
    m_preprocessedImgPub->publish(*msg);
    RCLCPP_DEBUG(this->get_logger(), "%zu images pré-traitées envoyées.", cvImages.size());

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