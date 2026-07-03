#pragma once // Optionnel mais fortement recommandé si c'est un .hpp

// --- STL (Bibliothèque standard C++) ---
#include <memory>  // Pour std::shared_ptr et std::make_shared
#include <vector>  // Pour std::vector (le stockage de tes publishers et topics)
#include <string>  // Pour std::string et std::to_string
#include <mutex>   // Pour std::mutex et std::lock_guard (sécurisation du buffer d'images)
#include <chrono>  // Pour std::chrono::milliseconds (la configuration du Timer à 20Hz)

// --- ROS 2 Core ---
#include "rclcpp/rclcpp.hpp"

// --- Messages ROS 2 Standard ---
#include "sensor_msgs/msg/image.hpp"  // Pour le type de message sensor_msgs::msg::Image
#include "std_msgs/msg/header.hpp"    // Pour manipuler le header (stamp et frame_id)

// --- OpenCV & cv_bridge (Conversion Vision) ---
#include "cv_bridge/cv_bridge.hpp"       // Pour faire la passerelle entre ROS 2 et OpenCV
#include <opencv2/opencv.hpp>          // Pour cv::Mat, cv::resize, cv::Rect, etc.

class cameraNode : public rclcpp::Node {
    public : 

    cameraNode();

    private :

    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg, const std::string& topic_name);

    void timerCallback();

    std::vector<std::string> m_cameraTopics;

    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr m_preprocessedImgPub;

    rclcpp::TimerBase::SharedPtr m_timer;

    std::mutex m_bufferMutex;

    std::map<std::string, sensor_msgs::msg::Image::SharedPtr> m_imageBuffer;

    std::vector<rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr> m_subscriptions;

};