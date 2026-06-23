#!/usr/bin/env python3
"""
Magdalena – Inference Node ROS 2 (YOLO classify via Ultralytics)
-------------------------------------------------------------------
One and multiples cameras

"""
 
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import Bool
 
import cv2
from ultralytics import YOLO
 
# ──────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────
 
# ── 1 caméra ──
CAMERA_TOPICS = [
    "/camera/camera/color/image_raw",
]
 
# ── 5 caméras
# CAMERA_TOPICS = [
#     "/camera_55/color/image_raw",
#     "/camera_56/color/image_raw",
#     "/camera_57/color/image_raw",
#     "/camera_58/color/image_raw",
#     "/camera_59/color/image_raw",
# ]
 
ENGINE_PATH    = "runs/classify/magdalena/plant_cls_v4/weights/best.engine"
TIMER_HZ       = 15.0   # Ips
 
# ──────────────────────────────────────────────────────────────
#  NODE
# ──────────────────────────────────────────────────────────────
 
class MagdalenaInferenceNode(Node):
 
    def __init__(self):
        super().__init__("magdalena_inference_node")
        self.bridge = CvBridge()
 
        # Load model .engine
        self.get_logger().info(f"Chargement : {ENGINE_PATH}")
        self.model = YOLO(ENGINE_PATH, task="classify")
        self.get_logger().info("Modèle prêt.")
 
        # Buffer : Last RGB frames received by the topics
        self.latest_frames: dict[str, cv2.typing.MatLike | None] = {
            t: None for t in CAMERA_TOPICS
        }
 
        # Subscription
        for topic in CAMERA_TOPICS:
            self.create_subscription(
                Image, topic,
                lambda msg, t=topic: self._image_cb(msg, t),
                qos_profile=10,
            )
        #Publisher
        self.valve_pubs = {}
        for topic in CAMERA_TOPICS: 
            cam_id = topic.split("/")[1]          
            pub_topic = f"/magdalena/{cam_id}/valve"
            self.valve_pubs[cam_id]=self.create_publisher(Bool, pub_topic, 10)
            self.get_logger().info(f"Publisher vanne : {pub_topic}")
 
        # Timer
        self.create_timer(1.0 / TIMER_HZ, self._inference_loop)
 
        self.get_logger().info(
            f"Nœud démarré | {len(CAMERA_TOPICS)} caméra(s)"
        )
 
    # ── Callback image ────────────────────────────────────────
 
    def _image_cb(self, msg: Image, topic: str):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            h,w, _ = frame.shape
            x_start = w//2 - 48
            x_end = w//2 + 48
            y_start = h//2 - 160
            y_end = h//2 +160

            frame = frame[y_start:y_end, x_start:x_end]
            self.latest_frames[topic] = frame

        except Exception as e:
            self.get_logger().error(f"[{topic}] cv_bridge: {e}")
 
    # ── Inference loop───────────────────────────────────
 
    def _inference_loop(self):
        frames = [self.latest_frames[t] for t in CAMERA_TOPICS]
 
        # wait frames from all the cameras
        if any(f is None for f in frames):
            return
 
        # Inférence batch
        results = self.model(
            frames,
            imgsz=(96, 320),   
            verbose=False,
        )
        #Analysing the results for each camera
        for topic, result in zip(CAMERA_TOPICS, results):
            cam_id   = topic.split("/")[1]          
            top1_idx = result.probs.top1            
            top1_conf = float(result.probs.top1conf)
            cls_name = result.names[top1_idx]
            msg_valve = Bool()

            if cls_name == "plant" : 
                msg_valve.data = True
            elif cls_name =="no_plant" : 
                msg_valve.data = False

            self.valve_pubs[cam_id].publish(msg_valve)

            self.get_logger().info(
                    f"[{cam_id}] → {cls_name} ({top1_conf:.2f})"
                )
            img_cv = self.latest_frames[topic].copy()
            cv2.putText(img_cv,f"{cls_name}",(10, 30),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 0, 255),2)
            img_cv = cv2.resize(img_cv, (640, 192)) 
            cv2.imshow("prediction", img_cv)
        cv2.waitKey(1) 
        
 
 
# ──────────────────────────────────────────────────────────────
# Main 
# ──────────────────────────────────────────────────────────────
 
def main(args=None):
    rclpy.init(args=args)
    node = MagdalenaInferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()
 
 
if __name__ == "__main__":
    main()
