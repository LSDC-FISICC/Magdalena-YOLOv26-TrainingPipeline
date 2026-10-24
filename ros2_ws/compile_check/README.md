# Compile check without the Jetson

Builds `cameraNode` completely and compiles the inference node (`main.cpp`,
`InferenceNode_v2.cpp`, `model.cpp`) against stub `NvInfer.h` /
`cuda_runtime_api.h` headers, so the C++ can be checked on any x86 machine
with ROS 2 Humble but without CUDA or TensorRT. `runtime_test.py` then runs
`cameraNode` with two simulated cameras and `tools/decision_logger.py`, and
prints the preprocessed batch (shape, encoding, value range, `topic@capture_ns`
frame ids) and the logged rows.

ROS 2 Humble without apt (e.g. on Ubuntu 24.04) via RoboStack:

```bash
micromamba create -n ros_humble -c robostack-staging -c conda-forge python=3.11 \
    ros-humble-rclcpp ros-humble-rclpy ros-humble-sensor-msgs ros-humble-std-msgs \
    ros-humble-diagnostic-msgs ros-humble-cv-bridge ros-humble-ament-cmake compilers cmake make pkg-config
micromamba run -n ros_humble cmake -S ros2_ws/compile_check -B ros2_ws/compile_check/build
micromamba run -n ros_humble cmake --build ros2_ws/compile_check/build -j2
micromamba run -n ros_humble python ros2_ws/compile_check/runtime_test.py
```

The real build (TensorRT, CUDA) is still `colcon build` on the Jetson.
