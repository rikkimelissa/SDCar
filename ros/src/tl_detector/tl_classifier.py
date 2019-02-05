#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
import tensorflow as tf
import numpy as np
import matplotlib
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

class TLClassifier(object):
    def __init__(self):

        rospy.init_node('tl_classifier')

        PATH_TO_MODEL = 'frozen_inference_graph.pb'
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            # Works up to here.
            with tf.gfile.GFile(PATH_TO_MODEL, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            self.d_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            self.d_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.d_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_d = self.detection_graph.get_tensor_by_name('num_detections:0')
        self.sess = tf.Session(graph=self.detection_graph)

        self.bridge = CvBridge()
        sub1 = rospy.Subscriber('/image_color', Image, self.image_cb)
        self.camera_image = None
        self.light_pub = rospy.Publisher('/tl_classifier', Int32, queue_size=1)
        self.has_image = False

        print('Classifier ready to go')

        self.loop()

    def loop(self):
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            if (self.has_image):
                cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
                self.get_classification(cv_image)
            rate.sleep()

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg

    def get_classification(self, img):
        # Bounding Box Detection.
        with self.detection_graph.as_default():
            # Expand dimension since the model expects image to have shape [1, None, None, 3].
            img_expanded = np.expand_dims(img, axis=0)  
            (boxes, scores, classes, num) = self.sess.run(
                [self.d_boxes, self.d_scores, self.d_classes, self.num_d],
                feed_dict={self.image_tensor: img_expanded})
#         return boxes, scores, classes, num

        list_ind = np.where(classes[0]==10)
        if len(list_ind[0]) > 0:
            ind = list_ind[0][0]

            ymin = boxes[0][ind][0] 
            xmin = boxes[0][ind][1]
            ymax = boxes[0][ind][2]
            xmax = boxes[0][ind][3]

            im_height = 600
            im_width = 800
            (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                          ymin * im_height, ymax * im_height)

            imageh = matplotlib.colors.rgb_to_hsv(img)

            tl = imageh[int(top):int(bottom),int(left):int(right),2]

            on = np.where(tl<200)
            tl[on] = 0
            (ymean, xmean) = np.median(np.transpose(np.nonzero(tl)),0)
            height = bottom - top

            if (ymean < height/3):
                tcolor = 0
            elif (ymean > 2.0/3*height):
                tcolor = 2
            else:
                tcolor = 0

        else:
            tcolor = 2

        self.light_pub.publish(tcolor)

        # return tcolor

if __name__ == '__main__':
    try:
        TLClassifier()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')




# from PIL import Image
#from tl_classifier import TrafficLightClassifier
# tl = TrafficLightClassifier()
# image = Image.open('0003.jpg')
# tl.get_classification(image)


