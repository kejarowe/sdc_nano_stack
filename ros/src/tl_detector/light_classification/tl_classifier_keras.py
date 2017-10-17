
from styx_msgs.msg import TrafficLight

import os
import cv2
import numpy as np

import tensorflow as tf
from keras.models import load_model


MODEL_FILE = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'model/model.h5')
assert os.path.exists(MODEL_FILE)


class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        self.model = load_model(MODEL_FILE)
        self.model.summary()
        # An issue: https://github.com/fchollet/keras/issues/2397
        self.graph = tf.get_default_graph()
    
       
    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        image = np.array([cv2.resize(image, (32, 32))])

        with self.graph.as_default():
            results = self.model.predict(image)

        matched_light = np.argmax(results[0])

        if matched_light == 0:
            return TrafficLight.GREEN
        if matched_light == 1:
            return TrafficLight.RED
        if matched_light == 2:
            return TrafficLight.YELLOW
        
#if __name__ == "__main__":
#    get_classification()
