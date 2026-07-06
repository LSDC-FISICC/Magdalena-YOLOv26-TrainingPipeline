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
#include <sstream>


class InferenceNode_v2 : public rclcpp::Node {
public:
    InferenceNode_v2();

private:
    // Callback for each batch and publish the prediction
    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg);

    // Class model
    std::unique_ptr<model> m_model;

    // Subcription
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr m_sharedImg;

    //To save the received batch with all the information (timestamp, camera_Id, coding)
    //std::map<rclcpp::Time, std::vector<std::pair<std::string, cv::Mat>>> m_batchCollector;
    size_t m_expectedBatchSize = 1;

    //Publisher

    rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr m_batchPlantPub;


};