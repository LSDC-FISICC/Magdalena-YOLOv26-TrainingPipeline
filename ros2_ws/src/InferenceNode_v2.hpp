#pragma once

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <vector>
#include <string>
#include <map>
#include <mutex>
#include <memory>
#include "model.hpp"
#include "diagnostic_msgs/msg/diagnostic_array.hpp"


class InferenceNode_v2 : public rclcpp::Node {
public:
    InferenceNode_v2();

private:
    // Callback générique appelé par CHAQUE caméra
    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg);

    // Instance de ta classe modèle
    std::unique_ptr<model> m_model;

    // Liste des abonnements ROS 2 générés dynamiquement
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr m_sharedImg;
    std::map<rclcpp::Time, std::vector<std::pair<std::string, cv::Mat>>> m_batchCollector;
    size_t m_expectedBatchSize = 1;

    //publisher

    rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr m_batchPlantPub;


};