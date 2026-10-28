#!/usr/bin/env python3
"""ROS 2 node that logs the per-camera decisions of both perception pipelines during field passes.

Valve control is done by modbus_controller (Magdalena-CV-Pipeline). This node
only records, on the Jetson clock:
  * YOLO26 decisions from /detection/Trigger (one row per camera and batch) with
    the stamps added by cameraNode and inference_node -> e7 latency, e8 savings,
  * colour/NDVI index decisions from /threshold1../thresholdN (Float32, the
    Magdalena-CV-Pipeline main branch), method "index", when log_index is true,
  * every ``snapshot_every`` batches, the 320x96 crops of /camera/preprocessed_frames
    as PNG, named <camera>_<capture_ns>.png, to build the field ground truth
    (e1_gt_sample.py --source <snapshot_dir>) including night passes.
Running both pipelines on the same pass (one drives the valves, the other is
only logged) gives the paired field comparison of e9_field_ab.py.

Run:
  python3 ros2_ws/tools/decision_logger.py --ros-args \
      -p log_csv:=/home/jetson/logs/decisions_$(date +%s).csv \
      -p log_index:=true -p index_threshold:=0.02 -p snapshot_dir:=/home/jetson/logs/frames -p snapshot_every:=15
"""

import csv
import os

import rclpy
from diagnostic_msgs.msg import DiagnosticArray
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32

FIELDS = ["method", "camera", "index", "raw", "score", "confidence", "t_capture_ns", "t_preprocessed_ns",
          "t_received_ns", "t_infer_done_ns", "t_logger_rx_ns",
          "blur_ratio", "blur_median", "model", "plant_main", "plant_fallback"]


class DecisionLogger(Node):
    def __init__(self):
        super().__init__("decision_logger")
        log_csv = self.declare_parameter("log_csv", "decisions.csv").value
        log_index = self.declare_parameter("log_index", False).value
        n_index = self.declare_parameter("index_topics", 5).value
        self.index_threshold = self.declare_parameter("index_threshold", 0.02).value
        self.snapshot_dir = self.declare_parameter("snapshot_dir", "").value
        self.snapshot_every = self.declare_parameter("snapshot_every", 15).value
        os.makedirs(os.path.dirname(os.path.abspath(log_csv)), exist_ok=True)
        self.fh = open(log_csv, "w", newline="")
        self.writer = csv.DictWriter(self.fh, fieldnames=FIELDS)
        self.writer.writeheader()
        self.create_subscription(DiagnosticArray, "/detection/Trigger", self.cb_yolo, 10)
        if log_index:
            for i in range(n_index):
                self.create_subscription(Float32, f"/threshold{i + 1}",
                                         lambda m, i=i: self.cb_index(m, i), 10)
        self.batches = 0
        if self.snapshot_dir:
            os.makedirs(self.snapshot_dir, exist_ok=True)
            self.create_subscription(Image, "/camera/preprocessed_frames", self.cb_frames, 2)
        self.get_logger().info(f"decision_logger: {log_csv} index={log_index} snapshots={self.snapshot_dir or 'off'}")

    def cb_yolo(self, msg):
        t_rx = self.get_clock().now().nanoseconds
        for pos, status in enumerate(msg.status):
            kv = {v.key: v.value for v in status.values}
            self.writer.writerow({
                "method": "yolo", "camera": status.name, "index": pos,
                "raw": int(kv.get("plant_detected") == "true"), "confidence": kv.get("confidence", ""),
                "t_capture_ns": kv.get("t_capture_ns", ""), "t_preprocessed_ns": kv.get("t_preprocessed_ns", ""),
                "t_received_ns": kv.get("t_received_ns", ""), "t_infer_done_ns": kv.get("t_infer_done_ns", ""),
                "t_logger_rx_ns": t_rx,
                **{k: kv.get(k, "") for k in ("blur_ratio", "blur_median", "model", "plant_main", "plant_fallback")}})
        self.fh.flush()

    def cb_index(self, msg, i):
        # modbus_controller (main branch) opens the valve when the index average >= threshold
        self.writer.writerow({"method": "index", "camera": f"threshold{i + 1}", "index": i,
                              "raw": int(msg.data >= self.index_threshold), "score": msg.data,
                              "t_logger_rx_ns": self.get_clock().now().nanoseconds})

    def cb_frames(self, msg):
        self.batches += 1
        if self.batches % self.snapshot_every:
            return
        import cv2
        import numpy as np
        from cv_bridge import CvBridge
        img = CvBridge().imgmsg_to_cv2(msg, msg.encoding)
        if img.dtype != np.uint8:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        for i, item in enumerate(msg.header.frame_id.split(",")):
            cam, _, stamp = item.partition("@")
            crop = img[i * 96:(i + 1) * 96]
            name = f"{cam.strip('/').split('/')[0]}_{stamp or msg.header.stamp.sec}.png"
            cv2.imwrite(os.path.join(self.snapshot_dir, name), crop)


def main():
    rclpy.init()
    node = DecisionLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.fh.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
