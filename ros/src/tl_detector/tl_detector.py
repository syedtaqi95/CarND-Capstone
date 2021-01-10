#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
from scipy.spatial import KDTree
import numpy as np

STATE_COUNT_THRESHOLD = 3
MAX_LIGHT_DIST = 50 # Max distance of a TL from the car to be considered visible

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.base_waypoints = None
        self.waypoints_2d = None
        self.waypoint_tree = None
        self.camera_image = None
        
        self.lights_2d = []
        self.lights_states = []
        self.lights_tree = None

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        self.loop()

    def loop(self):
        rate = rospy.Rate(10) # 10Hz
        while not rospy.is_shutdown():
            # Get the index of the closest waypoint and state of the closest traffic light
            light_wp_idx, state = self.process_traffic_lights()

            # Only publish the wp if the light is red
            if state == TrafficLight.RED:
                self.upcoming_red_light_pub.publish(Int32(light_wp_idx))
            else:
                self.upcoming_red_light_pub.publish(Int32(-1))

            rate.sleep()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.base_waypoints = waypoints
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint 
                                in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        lights = msg.lights
        # Save the 2D pose and state of all traffic lights, and create a KDTree for quick searching
        self.lights_2d = [[light.pose.pose.position.x, light.pose.pose.position.y] for light in lights]
        self.lights_states = [light.state for light in lights]
        self.lights_tree = KDTree(self.lights_2d)


    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        # light_wp, state = self.process_traffic_lights()

        # '''
        # Publish upcoming red lights at camera frequency.
        # Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        # of times till we start using it. Otherwise the previous stable state is
        # used.
        # '''
        # if self.state != state:
        #     self.state_count = 0
        #     self.state = state
        # elif self.state_count >= STATE_COUNT_THRESHOLD:
        #     self.last_state = self.state
        #     light_wp = light_wp if state == TrafficLight.RED else -1
        #     self.last_wp = light_wp
        #     self.upcoming_red_light_pub.publish(Int32(light_wp))
        # else:
        #     self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        # self.state_count += 1

    def get_closest_waypoint_idx(self, x, y):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        #TODO implement
        return self.waypoint_tree.query([x,y], 1)[1]

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if(not self.has_image):
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        #Get classification
        return self.light_classifier.get_classification(cv_image)

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        if(self.pose):
            car_pose_x = self.pose.pose.position.x
            car_pose_y = self.pose.pose.position.y
            car_position_idx = self.get_closest_waypoint_idx(car_pose_x, car_pose_y)
            car_position_wp = self.waypoints_2d[car_position_idx]

            #TODO find the closest visible traffic light (if one exists)
            if self.lights_tree:
                # Closest light to the car
                closest_light_dist, closest_light_idx = self.lights_tree.query([car_pose_x, car_pose_y], 1)

                if closest_light_dist < MAX_LIGHT_DIST:
                    # Check if light is ahead of the vehicle
                    closest_coord = self.lights_2d[closest_light_idx]
                    prev_coord = self.lights_2d[closest_light_idx-1]
                    
                    cl_vect = np.array(closest_coord)
                    prev_vect = np.array(prev_coord)
                    pose_vect = np.array([car_pose_x, car_pose_y])

                    # if dot product is positive, closest light is behind vehicle
                    val = np.dot(cl_vect - prev_vect, pose_vect - cl_vect)
                    if val < 0:
                        # Find the index of the closest waypoint to the stop position of the light
                        closest_stop_line_coord = stop_line_positions[closest_light_idx]
                        light_waypoint_idx = self.get_closest_waypoint_idx(closest_stop_line_coord[0], closest_stop_line_coord[1])
                        light_state = self.lights_states[closest_light_idx]

                        # Debug print
                        # debug_str =  "\nTraffic Light Distance: " + str(closest_light_dist) + "\n"
                        # debug_str += "Traffic Light State: " + str(light_state) + "\n"
                        # debug_str += "Traffic Light Index: " + str(closest_light_idx) + "\n"
                        # rospy.loginfo(debug_str)
                        
                        return light_waypoint_idx, light_state
        
        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
