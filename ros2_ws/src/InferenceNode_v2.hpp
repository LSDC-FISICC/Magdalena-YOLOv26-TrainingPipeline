#pragma once

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
// cv_bridge.hpp exists from cv_bridge 3.3 (Iron); stock Humble (3.2.x) only ships cv_bridge.h
#if __has_include(<cv_bridge/cv_bridge.hpp>)
#include <cv_bridge/cv_bridge.hpp>
#else
#include <cv_bridge/cv_bridge.h>
#endif
#include <vector>
#include <string>
#include <map>
#include <deque>
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

    // Pass-condition switch. A second engine (parameter "fallback_engine_path", e.g. the model
    // trained on ExG labels) decides for a camera while its pass is blurred: the median, over
    // the last "blur_window" frames of that camera, of blurRatio() is below "blur_threshold".
    // Empty fallback_engine_path disables the switch (main engine only, as before).
    static float blurRatio(const cv::Mat& img);
    std::unique_ptr<model> m_fallback;
    double m_blurThreshold = 0.561;
    size_t m_blurWindow = 30;
    std::map<std::string, std::deque<float>> m_blurHistory;
    std::map<std::string, bool> m_onFallback;

    // Subcription
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr m_sharedImg;

    //To save the received batch with all the information (timestamp, camera_Id, coding)
    //std::map<rclcpp::Time, std::vector<std::pair<std::string, cv::Mat>>> m_batchCollector;
    // Max batch the engine accepts (one image per camera); parameter "max_batch_size"
    int m_expectedBatchSize = 5;

    //Publisher

    rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr m_batchPlantPub;


};