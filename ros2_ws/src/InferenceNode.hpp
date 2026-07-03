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

class InferenceNode : public rclcpp::Node {
public:
    InferenceNode();

private:
    // Callback générique appelé par CHAQUE caméra
    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg, const std::string& topic_name);

    // Timer qui s'exécute périodiquement pour regrouper les images et lancer l'inférence
    void timerCallback();

    // Instance de ta classe modèle
    std::unique_ptr<model> m_model;

    // Liste des abonnements ROS 2 générés dynamiquement
    std::vector<rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr> m_subscriptions;

    // Stockage des dernières images reçues par topic
    std::map<std::string, sensor_msgs::msg::Image::SharedPtr> m_imageBuffer;
    
    // Mutex pour éviter les conflits d'accès entre les callbacks des caméras et le timer
    std::mutex m_bufferMutex;

    // Timer ROS 2
    rclcpp::TimerBase::SharedPtr m_timer;

    // Topics des caméras (basé sur ton vecteur)
    std::vector<std::string> m_cameraTopics;
};