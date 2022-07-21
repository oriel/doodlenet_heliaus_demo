import cv2
import torch
import argparse
import time
from utils.build_data import *
from utils.module_list import *

import torchvision.transforms.functional as TF

from utils.camera_utils import frame_to_pil

import rclpy 
from rclpy.node import Node 
from sensor_msgs.msg import Image 
from cv_bridge import CvBridge 

from utils.camera_utils import undistort, K_th, D_th, K_rgb, D_rgb, H

class ImageAlign(Node):
  """
  Create an ImageAlign class, which is a subclass of the Node class.
  """
  def __init__(self, rgb_topic='source_images_rgb', lwir_topic='source_images_lwir', display_cv2=False, width=640, height=480):
    """
    Class constructor to set up the node
    """
    # Initiate the Node class's constructor and give it a name
    super().__init__('doodle_net_node')
       
    # Create the subscriber. This subscriber will receive an Image
    # from the video_frames topic. The queue size is 10 messages.
    self.subscription_rgb = self.create_subscription(
      Image, 
      rgb_topic, 
      self.listener_callback_rgb, 
      10)
    self.subscription_rgb # prevent unused variable warning

    self.subscription_lwir = self.create_subscription(
      Image, 
      lwir_topic, 
      self.listener_callback_lwir, 
      10)
    self.subscription_lwir # prevent unused variable warning
        
    # Used to convert between ROS and OpenCV images
    self.br = CvBridge()

    self.data_rgb = None
    self.data_lwir = None
    self.received_rgb = False
    self.received_lwir = False

    self.publisher_rgb = self.create_publisher(Image, 'aligned_rgb', 10)
    self.publisher_lwir = self.create_publisher(Image, 'aligned_lwir', 10)

    self.display_cv2 = display_cv2

    self.width = width
    self.height = height


  def listener_callback_rgb(self, data):
    """
    Callback function for rgb stream. 
    Perform undistortion + resize + crop + affine align
    """
    self.get_logger().info('Receiving RGB video frame')
    self.received_rgb = True
    self.data_rgb = data
    _, cv2_img = self.ros_to_tensor(data)
    cv2_img = undistort(cv2_img, K, D, fisheye=False)
    cv2_img = cv2.resize(cv2_img, (self.width, self.height))
    cv2_img = cv2.warpPerspective(cv2_img, H, (self.width, self.height))
    self.publisher_rgb.publish(self.br.cv2_to_imgmsg(cv2_img, encoding="rgb8"))
    self.get_logger().info('Publishing rgb aligned video frame output')

    

  def listener_callback_lwir(self, data):
    """
    Callback function for lwir stream. 
    Only perform undistortion.
    """
    self.get_logger().info('Receiving LWIR video frame')
    self.received_lwir = True
    _, cv2_img = self.ros_to_tensor(data)
    cv2_img = undistort(cv2_img, K_th, D_th, fisheye=True)
    self.publisher_lwir.publish(self.br.cv2_to_imgmsg(cv2_img, encoding="rgb8"))
    self.get_logger().info('Publishing lwir undistorted video frame output')

    
  def ros_to_tensor(self, data):
    """
    Takes as input a ROS2 image message data, returns a torch tensor and a opencv/numpy array.
    """
    cv_image = self.br.imgmsg_to_cv2(data, desired_encoding='passthrough')
    print("MSG_ENCODING: " + data.encoding)
    if data.encoding == 'mono16' or data.encoding =='mono8' :
       if data.encoding == 'mono16':
          cv_image = (cv_image/256).astype('uint8')
       cv_image_8bits = cv2.merge((cv_image,cv_image,cv_image))    
    if data.encoding == 'yuv422':
       cv_image_8bits =self.br.imgmsg_to_cv2(data, desired_encoding='rgb8')
    if cv_image_8bits.shape[0] > 480:
       cv_image_8bits = cv2.resize(cv_image_8bits,(640,480))
    
    x_pil = frame_to_pil(cv_image_8bits)
    tx = TF.to_tensor(x_pil).unsqueeze(0).to(self.device)
    
    return tx, cv_image_8bits
    
    
def get_args():
 
    parser = argparse.ArgumentParser(description='Perform rectification + alignment for AB Heliaus demo', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument('--rgb_topic', default="source_images_rgb", type=str, help='Name of RGB video stream ROS2 topic')
    parser.add_argument('--lwir_topic', default="source_images_lwir", type=str, help='Name of LWIR video stream ROS2 topic')
    parser.add_argument('--display', action='store_true', help='Display output frames with opencv (only works from local terminal supporting graphical server)')


    args = parser.parse_args()
    return args
        

def main(args):
    
    #############

    # Initialize the rclpy library
    rclpy.init(args=None)
   
    # Create the node
    print(f'Starting ROS2 subscriber')
    print(f'Waiting for image messages at topics {args.rgb_topic} , {args.lwir_topic}')

    image_aligner = ImageAlign(rgb_topic=args.rgb_topic, lwir_topic=args.lwir_topic, display_cv2=args.display)
   
    # Spin the node so the callback function is called.
    rclpy.spin(image_aligner)
   
    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    image_subscriber.destroy_node()
   
    # Shutdown the ROS client library for Python
    rclpy.shutdown()


if __name__ == '__main__':
    args = get_args()
    main(args)
    
