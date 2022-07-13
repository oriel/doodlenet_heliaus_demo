import cv2
import torch
import torchvision.models as models
import argparse
import time
from utils.build_data import *
from utils.module_list import *

from network.segmentation_models_pytorch import DeepLabV3PlusDoubleEncoder, DeepLabV3Plus

import torchvision.transforms.functional as TF

from utils.camera_utils import frame_to_pil

import rclpy 
from rclpy.node import Node 
from sensor_msgs.msg import Image 
from cv_bridge import CvBridge 

class ImageSubscriberInference(Node):
  """
  Create an ImageSubscriberInference class, which is a subclass of the Node class.
  """
  def __init__(self, model, device, colormap, rgb_topic='source_images_rgb', lwir_topic='source_images_lwir', display_cv2=False):
    """
    Class constructor to set up the node
    """
    # Initiate the Node class's constructor and give it a name
    super().__init__('image_subscriber')
       
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
    self.model = model
    self.device = device
    self.colormap = colormap

    self.data_rgb = None
    self.data_lwir = None
    self.received_rgb = False
    self.received_lwir = False

    self.publisher_ = self.create_publisher(Image, 'out_frames', 10)
    self.display_cv2 = display_cv2


  def listener_callback_rgb(self, data):
    """
    Callback function for rgb stream.
    """
    self.get_logger().info('Receiving RGB video frame')
    self.received_rgb = True
    self.data_rgb = data
    self.sync_streams()

  def listener_callback_lwir(self, data):
    """
    Callback function for lwir stream.
    """
    self.get_logger().info('Receiving LWIR video frame')
    self.received_lwir = True
    self.data_lwir = data
    self.sync_streams()
    
  def sync_streams(self):
    """
    Verify if RGB and LWIR images are ready, if so perform inference.
    """    
    # check if we have received both rgb and lwir data, otherwise wait
    if self.received_rgb and self.received_lwir:
      # reset flags
      self.received_rgb = False
      self.received_lwir = False
      # do inference
      self.inference(self.data_rgb, self.data_lwir)

  def ros_to_tensor(self, data):
    """
    Takes as input a ROS2 image message data, returns a torch tensor and a opencv/numpy array.
    """
    cv_frame = self.br.imgmsg_to_cv2(data)
    x_pil = frame_to_pil(cv_frame)
    tx = TF.to_tensor(x_pil).unsqueeze(0).to(self.device)
    return tx, cv_frame
    
  def inference(self, data_rgb, data_lwir):
    """
    Perform semantic segmentation inference with torch model.
    Takes as input a 3-channel RGB tensor and 1-channel LWIR tensor, concatenates it and returns a segmented image.
    """
    start_time = time.time()
      
    # Convert ROS Image message to torch and OpenCV image
    try:
      tx_rgb, cv_frame_rgb = self.ros_to_tensor(data_rgb)
      tx_lwir, cv_frame_lwir = self.ros_to_tensor(data_lwir)
    except:
      print('Error: not possible to read video stream')
      return None

    # thermal: 3 channels to 1 channel
    if tx_lwir.shape[1] > 1:
      tx_lwir = tx_lwir[:,0,:,:].unsqueeze(0).to(self.device)

    # concatenate RGB and LWIR
    txy = torch.cat((tx_rgb, tx_lwir), 1)

    # compute model predictions
    logits = self.model(txy)
    logits = F.interpolate(logits, size=tx_rgb.shape[2:], mode='bilinear', align_corners=True)
    prob, max_logits = torch.max(torch.softmax(logits, dim=1), dim=1)

    # apply color map to predictions
    predicted_frame = (color_map(max_logits[0].cpu().numpy(), self.colormap))
    predicted_frame = cv2.cvtColor(predicted_frame, cv2.COLOR_BGR2RGB)
    cv_frame = cv2.addWeighted(cv_frame_rgb, 0.5, cv_frame_lwir, 0.5, 0)
    out_frame = cv2.addWeighted(cv_frame, 0.5, predicted_frame, 0.5, 0)

    # publish result for external ros-based visualization
    self.publisher_.publish(self.br.cv2_to_imgmsg(out_frame))
    self.get_logger().info('Publishing video frame output')

    fps = int(1.0/(time.time() - start_time))
    print(f"Demo speed: {fps}")
     
    # Display image by opencv if asked
    if self.display_cv2:
        cv2.imshow(f"Demo", out_frame)     
        cv2.waitKey(1)

    
def get_args():
 
    parser = argparse.ArgumentParser(description='Perform semantic segmentation on heliaus test data, generates a video demo', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument("--checkpoint", type=str, default='model_weights/DeepLabV3PlusDoubleEncoder_confidence_mobilenet_debugged_seed_42_finetune_same_checkpoint.pth')
    parser.add_argument('--backbone', default="mobilenet_v2", type=str, help='Name of the backbone to be used with the model (resnet101, resnet34)')
    parser.add_argument('--num_classes', default=9, type=int, help='number of training data classes')
    parser.add_argument('--dataset', default='heliaus_rgbt', type=str, help='pascal, cityscapes, sun, kaist_rgbt, kaist_rgb, rtfnet_rgbt, rtfnet_rgb, rtfnet_rgbt, heliaus_rgb, heliaus_rgbt')
    parser.add_argument('--model', default="doodlenet", type=str, help='Name of model to be used (deeplab3, doodlenet)')
    parser.add_argument('--show_complexity', action='store_true', help='Show numbers of params and flops')
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument('--rgb_topic', default="source_images_rgb", type=str, help='Name of RGB video stream ROS2 topic')
    parser.add_argument('--lwir_topic', default="source_images_lwir", type=str, help='Name of LWIR video stream ROS2 topic')
    parser.add_argument('--display', action='store_true', help='Display output frames with opencv (only works from local terminal supporting graphical server)')


    args = parser.parse_args()
    return args
        

def main(args):
    
    ### load a sequential test loader based on asked sequence
    rgbt = 'rgbt' in args.dataset
    num_classes = args.num_classes

    device = torch.device("cuda:{:d}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    
    ### Load Semantic Network
    if args.model == 'deeplab3':
        model = DeepLabV3Plus(
            encoder_name=args.backbone, 
            encoder_weights="imagenet", 
            in_channels=4, 
            classes=args.num_classes)

    elif args.model == 'doodlenet':
        model = DeepLabV3PlusDoubleEncoder(
            encoder_name=args.backbone, 
            encoder_weights="imagenet", 
            in_channels=3, 
            classes=args.num_classes,
            correlation_weight=False,
            confidence_weight=True,
            multiloss=True)

    #### Load model weights from trained model
    print(f"Loaded {args.model} model")
    gpu=0
    pretrained_weight = torch.load(args.checkpoint, map_location = lambda storage, loc: storage.cuda(gpu))
    model.load_state_dict(pretrained_weight, strict=False)
    model.cuda(0)
    model.eval()

    #model.load_state_dict(torch.load(args.checkpoint, strict=False))
    print("Loaded model: {}".format(args.checkpoint))

    #### if asked, compute model flops and parameters, then exit
    if args.show_complexity:
        print_model_complexity(model, args.model, args.backbone, rgbt, device)
        exit()

    colormap = create_heliaus_label_colormap()

    #############

    # Initialize the rclpy library
    rclpy.init(args=None)
   
    # Create the node
    print(f'Starting ROS2 subscriber')
    print(f'Waiting for image messages at topics {args.rgb_topic} , {args.lwir_topic}')

    image_subscriber = ImageSubscriberInference(model, device, colormap, rgb_topic=args.rgb_topic, lwir_topic=args.lwir_topic, display_cv2=args.display)
   
    # Spin the node so the callback function is called.
    rclpy.spin(image_subscriber)
   
    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    image_subscriber.destroy_node()
   
    # Shutdown the ROS client library for Python
    rclpy.shutdown()


if __name__ == '__main__':
    args = get_args()
    main(args)
    
