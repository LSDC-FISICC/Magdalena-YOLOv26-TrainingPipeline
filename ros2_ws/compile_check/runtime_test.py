import csv, os, subprocess, sys, time
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from cv_bridge import CvBridge

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.join(HERE, "build")
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
LOG = os.path.join(BUILD, "decisions_test.csv")
cam = subprocess.Popen([os.path.join(BUILD, "cameraNode"), "--ros-args",
                        "-p", "camera_topics:=['/camera_55/color/image_raw','/camera_56/color/image_raw']"],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
logger = subprocess.Popen([sys.executable, os.path.join(REPO, "ros2_ws", "tools", "decision_logger.py"),
                           "--ros-args", "-p", f"log_csv:={LOG}", "-p", "log_index:=true"],
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
rclpy.init()
n = Node("tester")
br = CvBridge()
pubs = [n.create_publisher(Image, f"/camera_5{i}/color/image_raw", 10) for i in (5, 6)]
trig = n.create_publisher(DiagnosticArray, "/detection/Trigger", 10)
got = []
n.create_subscription(Image, "/camera/preprocessed_frames", lambda m: got.append(m), 10)
time.sleep(2.0)
t_end = time.time() + 4
while time.time() < t_end:
    for k, p in enumerate(pubs):
        img = np.full((360, 640, 3), 60 + 100 * k, np.uint8)
        m = br.cv2_to_imgmsg(img, "bgr8"); m.header.stamp = n.get_clock().now().to_msg()
        p.publish(m)
    rclpy.spin_once(n, timeout_sec=0.06)
# emulate one inference result so the logger writes a row
arr = DiagnosticArray()
for cam_name, val in (("/camera_55/color/image_raw", "true"), ("/camera_56/color/image_raw", "false")):
    st = DiagnosticStatus(name=cam_name)
    st.values = [KeyValue(key="plant_detected", value=val), KeyValue(key="confidence", value="0.9"),
                 KeyValue(key="t_capture_ns", value="1"), KeyValue(key="t_preprocessed_ns", value="2"),
                 KeyValue(key="t_received_ns", value="3"), KeyValue(key="t_infer_done_ns", value="4")]
    arr.status.append(st)
for _ in range(3):
    trig.publish(arr); rclpy.spin_once(n, timeout_sec=0.2)
time.sleep(1)
cam.terminate(); logger.terminate()
print("preprocessed batches received:", len(got))
if got:
    m = got[-1]; img = br.imgmsg_to_cv2(m, m.encoding)
    print("encoding:", m.encoding, "shape:", img.shape, "value range:", float(img.min()), float(img.max()))
    print("frame_id:", m.header.frame_id[:160])
rows = list(csv.DictReader(open(LOG))) if os.path.exists(LOG) else []
print("logger rows:", len(rows), rows[:2])
rclpy.shutdown()
