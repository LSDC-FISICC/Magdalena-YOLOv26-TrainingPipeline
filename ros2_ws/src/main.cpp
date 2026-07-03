#include "InferenceNode_v2.hpp"


int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<InferenceNode_v2>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}