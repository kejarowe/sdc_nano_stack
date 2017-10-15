
from styx_msgs.msg import TrafficLight

import os
import cv2
import numpy as np

import tensorflow as tf


MODEL_FILE = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'model/model.pb')
assert os.path.exists(MODEL_FILE)


class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        self.graph = self.load_graph(MODEL_FILE)

    def load_graph(self, model_abs_path):
        graph = tf.Graph()
        graph_def = tf.GraphDef()
        print(model_abs_path)
        with open(model_abs_path, "rb") as f:
          graph_def.ParseFromString(f.read())
        with graph.as_default():
          tf.import_graph_def(graph_def)

        return graph
    
       
    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        image = np.array([cv2.resize(image, (32, 32))])

        input_name = 'import/features'
        output_name = 'import/predicts/Softmax'
	input_operation = self.graph.get_operation_by_name(input_name)
        output_operation = self.graph.get_operation_by_name(output_name)
        learn_phase = self.graph.get_operation_by_name('import/fc1_drop/keras_learning_phase')

	with tf.Session(graph=self.graph) as sess:
	    results = sess.run(output_operation.outputs[0],
		              {input_operation.outputs[0]: image,
                               learn_phase.outputs[0]: False})

        matched_light = np.argmax(results[0])

        if matched_light == 0:
            return TrafficLight.GREEN
        if matched_light == 1:
            return TrafficLight.RED
        if matched_light == 2:
            return TrafficLight.YELLOW
        
#if __name__ == "__main__":
#    get_classification()
