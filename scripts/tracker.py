
#!/usr/bin/env python3
import cv2
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist, PoseStamped
from geometry_msgs.msg import Vector3,TwistWithCovariance, PoseWithCovariance,PoseArray,Twist
import tf
import time
import math
from sensor_msgs.msg import LaserScan
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import random
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
from nav_msgs.msg import Path, Odometry

import rospy


class YoloTracker:
    def __init__(self, weights_path, config_path, classes_path):
        self.net = cv2.dnn.readNet(weights_path, config_path)
        self.classes = []
        self.tracker = cv2.TrackerCSRT_create()
        self.bbox = None
        self.tracking = False
        self.bridge = CvBridge()
        self.u=None
        self.v=None
        self.z=None
        self.result=[]
        with open(classes_path, "r") as f:
            self.classes = [line.strip() for line in f]

        self.layer_names = self.net.getUnconnectedOutLayersNames()
        rospy.init_node('yolo')
        self.sub = rospy.Subscriber("/camera/rgb/image_raw",Image, self.image_callback)
        self.sub_depth = rospy.Subscriber('/camera/depth/image_raw',Image,self.depth_callback)
####OBSTACLE AVOIDANCE PARAMETERs###
        self.current_position = Odometry()
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.scan_sub=rospy.Subscriber('/scan', LaserScan, self.callback)
        self.obstacles=[2.0,2.0]
        self.listener = tf.TransformListener()

        # Wait for the transformation to become available
    
        # PID parameters
        self.Kp_linear = 0.12
        self.Ki_linear = 0.00000
        self.Kd_linear = 0.01
        self.Kp_angular = 0.08
        self.Ki_angular = 0.0
        self.Kd_angular = 0.08
        self.cmd_vel_msg = Twist()
        self.Rd=1.8
        # PID variables
        self.last_time = 0.0
        self.linear_errors_sum = 0.0
        self.angular_errors_sum = 0.0
        self.linear_previous_error = 0.0
        self.angular_previous_error = 0.0
        self.goal_positions=[3,0]
        self.current_goal=0
        self.obstacles=[2.0,2.0]
        self.goal_update_rate = rospy.Rate(0.2)  # Update goal every 2 seconds
        self.last_goal_update_time = rospy.get_time()
        self.time_values = []
        self.linear_error_values = []
        self.linear_output_values = []

    def callback(self,data):

        global obstacle_points 
        # Use the global list to store the points
        self.listener.waitForTransform('base_footprint', 'odom', rospy.Time(), rospy.Duration(4.0))

        # Clear the previous points
        lidar_frame = 'base_footprint'

        obstacle_points=[]

        for i, range_value in enumerate(data.ranges):
            # print(range_value)
            if math.isfinite(range_value):
                
                # Calculate angle in radians
                angle = data.angle_min + i * data.angle_increment

                # Convert polar coordinates to Cartesian coordinates
                x = range_value * math.cos(angle)
                y = range_value * math.sin(angle)

                # Create a PointStamped message with the LIDAR coordinates
                lidar_point = PointStamped()
                lidar_point.header.frame_id = lidar_frame
                lidar_point.point.x = x
                lidar_point.point.y = y
                lidar_point.point.z = 0.0  # Assuming the LIDAR is in the xy plane

                # Transform the LIDAR coordinates to the global frame ('odom' frame)
        
                global_point = self.listener.transformPoint('odom', lidar_point)
                # Now you have the obstacle coordinates in the global frame
                global_coordinates = global_point.point
                obstacle_points.append((global_coordinates.x,global_coordinates.y))

     

        obstacle_points = np.array([point for point in obstacle_points if abs(point[0]) <= 8 and abs(point[1]) <= 8])
        # print(obstacle_points)
        self.center=np.array([])
        self.center = np.mean(obstacle_points, axis=0)
          # Use mean, or np.median for a more robust estimate
        # assert obstacle_points.shape[0] == self.center.shape[0], "Shapes are not compatible."
        # Estimate the radius of the circular region (using the maximum distance to the center)
        if self.center.size!=0:
            distances = np.linalg.norm(obstacle_points - self.center, axis=1)
            radius = np.max(distances)+1.5
            self.Rd= radius

            # print(self.center)

            # Define the center and radius of the circle
            circle_center = (self.center[0], self.center[1])
            circle_radius = radius


            # Add the circle to the axis
            circle = Circle(circle_center, circle_radius, color='r', fill=False)


        # Set the limits for x and y
    
        # Plot the current position of the robot
        robot_x = self.current_position.pose.pose.position.x
        robot_y = self.current_position.pose.pose.position.y
        if min(data.ranges)<1:
            print("Obstacle Detected")
            print(self.result)
            self.goal = [self.result[0][2]/1000, -self.result[0][0]/1000]
            self.pid_control()
        else:
            print("NO obstacle detected")

    
    def attractive_force(self):
        k_att=50
        rob_x=self.current_position.pose.pose.position.x
        rob_y=self.current_position.pose.pose.position.y
        rg=math.sqrt((self.current_position.pose.pose.position.x-self.goal[0])**2+(self.current_position.pose.pose.position.y-self.goal[1])**2)
        if rg<0.5:
            self.cmd_vel_msg.linear.x=0
            self.cmd_vel_msg.linear.y=0
            self.cmd_vel_msg.angular.z=0
        F_att=[0,0]
        F_att[0]=-k_att*(rob_x-self.goal[0])/rg
        F_att[1]=-k_att*(rob_y-self.goal[1])/rg

        return F_att
    
    def repulsive_force(self):
        k_i=20
        rob_x=self.current_position.pose.pose.position.x
        rob_y=self.current_position.pose.pose.position.y
        empty=[0,0]
        if math.isfinite(self.center[0]):
            ro=math.sqrt((self.current_position.pose.pose.position.x-self.center[0])**2+(self.current_position.pose.pose.position.y-self.center[1])**2)
            
            if ro<=self.Rd:

                F_rep=[0,0]
                F_rep[0]=k_i*(rob_x-self.center[0])/(ro**3)
                F_rep[1]=k_i*(rob_y-self.center[1])/(ro**3)

                return F_rep
        return empty
    


    def odom_callback(self, odom):
        self.current_position = odom
        

    def pid_control(self):
        rate = rospy.Rate(10)  # 10 Hz update rate
        while not rospy.is_shutdown():
            F_att= self.attractive_force()
            F_rep= self.repulsive_force()
            F_x= F_att[0]+F_rep[0]
            F_y= F_att[1]+F_rep[1]
            alpha=math.atan2(F_y,F_x)

            # Linear PID control
            linear_error_x = self.goal[0] - self.current_position.pose.pose.position.x
            linear_error_y = self.goal[1] - self.current_position.pose.pose.position.y
            linear_error = math.sqrt(linear_error_x**2 + linear_error_y**2)

            linear_current_time = time.time()
            linear_delta_time = (linear_current_time - self.last_time)

            linear_errors_derivative = (linear_error - self.linear_previous_error) / linear_delta_time
            self.linear_errors_sum += linear_error * linear_delta_time

            linear_output = (
                self.Kp_linear * linear_error +
                self.Ki_linear * self.linear_errors_sum +
                self.Kd_linear * linear_errors_derivative
            )
            self.time_values.append(time.time())
            self.linear_error_values.append(linear_error)
            self.linear_output_values.append(linear_output)

            # Angular PID control
            desired_yaw = alpha
            current_yaw = 2 * math.atan2(
                self.current_position.pose.pose.orientation.z,
                self.current_position.pose.pose.orientation.w
            )

            angular_error = desired_yaw - current_yaw

            angular_current_time = time.time()
            angular_delta_time = (angular_current_time - self.last_time)

            angular_errors_derivative = (angular_error - self.angular_previous_error) / angular_delta_time
            self.angular_errors_sum += angular_error * angular_delta_time

            angular_output = (
                self.Kp_angular * angular_error +
                self.Ki_angular * self.angular_errors_sum +
                self.Kd_angular * angular_errors_derivative
            )

            # Update previous values
            self.linear_previous_error = linear_error
            self.angular_previous_error = angular_error
            self.last_time = time.time()


            # Publish Twist message to control the TurtleBot
            if abs(linear_error)<0.6 and  abs(angular_error)<0.2:
                
                self.cmd_vel_msg.linear.x = 0
                self.cmd_vel_msg.angular.z = 0
                break
            elif abs(linear_error)>0.6 and abs(angular_error)<0.2:
                # print("angular stop")
                self.cmd_vel_msg.linear.x = linear_output
                self.cmd_vel_msg.angular.z = 0
            elif abs(linear_error)<0.6 and abs(angular_error)>0.2:
                # print("linear stop")
                self.cmd_vel_msg.linear.x = 0
                self.cmd_vel_msg.angular.z = angular_output
            else:
                self.cmd_vel_msg.linear.x = linear_output
                self.cmd_vel_msg.angular.z = angular_output

            self.cmd_vel_pub.publish(self.cmd_vel_msg)
            rate.sleep()


        
    def depth_callback(self,data):
        dimage=data
        bridge = CvBridge()
        depth_image=bridge.imgmsg_to_cv2(dimage, desired_encoding="passthrough")
        self.z= depth_image[self.v][self.u]*1000
        if np.isnan(self.z).any():
            self.z = 3000

    def image_callback(self, msg):
        print("Following the hooman")
        prop_linear = 3
        prop_angular = 0.1
        cmd_vel_msg = Twist()
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)
            return
        
        
        if not self.tracking:
            # Object detection using YOLOv3
            height, width, _ = cv_image.shape
            blob = cv2.dnn.blobFromImage(cv_image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            self.net.setInput(blob)
            outs = self.net.forward(self.layer_names)
            
            # Get bounding box and confidence for each detection
            class_ids = []
            confidences = []
            boxes = []

            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]

                    if confidence > 0.95:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            # Apply non-maximum suppression to eliminate redundant overlapping boxes
            indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.95, 0.8)

            if len(indices) > 0:
                i = indices[0]
                self.bbox = tuple(boxes[i])
                self.tracking = True
                self.tracker.init(cv_image, self.bbox)

        else:
            # Object tracking using KCF
            self.tracking, self.bbox = self.tracker.update(cv_image)

            if self.tracking:
                height, width, _ = cv_image.shape
                mid = width/2
                vel_angular = 0
                vel_linear = 0
                x, y, w, h = [int(val) for val in self.bbox]
                self.u=(x+ x + w)//2
                self.v=(y+ y + h)//2
                self.camera_to_world_coordinates()

                coordinates_text = f'({self.u}, {self.v},{self.z})'

        # Define the position where you want to place the text
                position = (self.u, self.v)

                # Define the font settings
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                font_color = (0, 0, 255)  # Red color in BGR format
                font_thickness = 1
                # Add the coordinates as text to the image
                cv2.putText(cv_image, coordinates_text, position, font, font_scale, font_color, font_thickness)
                cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                if(mid - (x+w) > 0):
                    vel_linear = prop_linear * math.tanh((self.z - 2200)/10000)/5
                    vel_angular = prop_angular * math.tanh(mid - (x+w))
                elif(mid - x < 0):
                    vel_linear = prop_linear * math.tanh((self.z - 2200)/10000)/5
                    vel_angular = prop_angular * (math.tanh(mid - x))
                elif(mid - x > 0 and mid - (x+w) < 0):
                    vel_angular = 0
                    vel_linear = 0.10
                    if(np.isnan(self.z).any() == False):
                        vel_linear = prop_linear * math.tanh((self.z - 2000)/10000)
                    else:
                        vel_linear *= 0.5
                        
                
                cmd_vel_msg.linear.x = vel_linear
                cmd_vel_msg.angular.z = vel_angular
                self.cmd_vel_pub.publish(cmd_vel_msg)
            else:
                cmd_vel_msg.linear.x = 0.001
                cmd_vel_msg.angular.z = 0
                self.cmd_vel_pub.publish(cmd_vel_msg)
        cv_image = cv2.resize(cv_image, dsize=(1080, 640))
        cv2.imshow('Tracking', cv_image)
        cv2.waitKey(1)  

    
    def camera_to_world_coordinates(self):
        flat_P=np.array([1206.8897719532354, 0.0, 960.5, -84.48228403672648, 0.0, 1206.8897719532354, 540.5, 0.0, 0.0, 0.0, 1.0, 0.0])
        P = np.reshape(flat_P,(3,4))
        points = [float(self.u), float(self.v)]
        K= [1206.8897719532354, 0.0, 960.5, 0.0, 1206.8897719532354, 540.5, 0.0, 0.0, 1.0]
        K=np.reshape(K,(3,3))
        intrinsic=K
        f_x = intrinsic[0, 0]
        f_y = intrinsic[1, 1]
        c_x = intrinsic[0, 2]
        c_y = intrinsic[1, 2]
        # This was an error before
        # c_x = intrinsic[0, 3]
        # c_y = intrinsic[1, 3]
        distortion = np.array([0.0, 0.0, 0.0, 0.0])
        Z=np.array([self.z])
        # Step 1. Undistort.
        points_undistorted = np.array([])
        if len(points) > 0:
            points_undistorted = cv2.undistortPoints(np.expand_dims(points, axis=1), intrinsic, distortion, P=P)
        points_undistorted = np.squeeze(points_undistorted, axis=1)

        # Step 2. Reproject.
        self.result = []
        for idx in range(points_undistorted.shape[0]):
            z = Z[0] if len(Z) == 1 else Z[idx]
            x = (points_undistorted[idx, 0] - c_x) / f_x * z
            y = (points_undistorted[idx, 1] - c_y) / f_y * z
            self.result.append([x, y, z])


    def run(self):
        rospy.spin()
        return self.result[0], self.result[1]

if __name__ == '__main__':
    weights_path = "yolov3.weights"
    config_path = "yolov3.cfg"
    classes_path = "yolov3.txt"
    image_topic = "/camera/rgb/image_raw"  # Replace with your actual ROS image topic

    yolo_tracker = YoloTracker(weights_path, config_path, classes_path)
    yolo_tracker.run()
    
