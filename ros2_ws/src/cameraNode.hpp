#pragma once

#include <memory>  //  std::shared_ptr et std::make_shared
#include <vector>  //  std::vector 
#include <string>  //  std::string et std::to_string
#include <mutex>   //  std::mutex et std::lock_guard 
#include <chrono>  //  std::chrono::milliseconds

// --- ROS 2 Core ---
#include "rclcpp/rclcpp.hpp"

// --- Messages ROS 2 Standard ---
#include "sensor_msgs/msg/image.hpp"  
#include "std_msgs/msg/header.hpp" 

// --- OpenCV & cv_bridge ---
// cv_bridge.hpp exists from cv_bridge 3.3 (Iron); stock Humble (3.2.x) only ships cv_bridge.h
#if __has_include(<cv_bridge/cv_bridge.hpp>)
#include <cv_bridge/cv_bridge.hpp>
#else
#include <cv_bridge/cv_bridge.h>
#endif      
#include <opencv2/opencv.hpp>          //cv::Mat, cv::resize, cv::Rect, etc.

class cameraNode : public rclcpp::Node {
    public : 

    cameraNode();

    private :

    //function to receive the images and to save them in m_imageBuffer
    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg, const std::string& topic_name);

    //function to process the images and publish them
    void timerCallback();

    //List of camera Topics for the subscription (parameter "camera_topics")
    std::vector<std::string> m_cameraTopics;

    //Crop window in the source frame, resized to 320x96 (parameters roi_x/roi_y/roi_w/roi_h).
    //roi_w/roi_h <= 0 keeps the default centred 320x96 crop.
    int m_roiX, m_roiY, m_roiW, m_roiH;

    //subscription
    std::vector<rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr> m_subscriptions;

    //publisher
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr m_preprocessedImgPub;

    //timer
    rclcpp::TimerBase::SharedPtr m_timer;

    //buffer to save the images from the cameras
    std::mutex m_bufferMutex;
    std::map<std::string, sensor_msgs::msg::Image::SharedPtr> m_imageBuffer;

    

};