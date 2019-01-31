import tensorflow as tf
import numpy as np
import matplotlib

class TrafficLightClassifier(object):
    def __init__(self):
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
        else:
            return 100

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
            tcolor = 1

        return tcolor





# from PIL import Image
#from tl_classifier import TrafficLightClassifier
# tl = TrafficLightClassifier()
# image = Image.open('0003.jpg')
# tl.get_classification(image)


