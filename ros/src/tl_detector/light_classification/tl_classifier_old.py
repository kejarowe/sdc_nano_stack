
# Based on tensorflow example label_image.py
from styx_msgs.msg import TrafficLight
import tensorflow as tf
import argparse
import sys
import numpy as np
import os

model_file = 'model/output_graph.pb'

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        self.graph = self.load_graph(model_file)
      
    
    def load_graph(self, model_file):
        graph = tf.Graph()
        graph_def = tf.GraphDef()
        model_abs_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), model_file)
        print(model_abs_path)
        with open(model_abs_path, "rb") as f:
          graph_def.ParseFromString(f.read())
        with graph.as_default():
          tf.import_graph_def(graph_def)

        return graph


    def read_tensor_from_image(self, image, input_height=224, input_width=224,
                input_mean=128, input_std=128):
        

        output_name = "normalized"
        float_caster = tf.cast(image, tf.float32)
        dims_expander = tf.expand_dims(float_caster, 0);
        resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
        normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
        sess = tf.Session()
        result = sess.run(normalized)

        return result
    
    
       
    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction

        input_name = "import/input"
        output_name = "import/final_result"

        input_operation = self.graph.get_operation_by_name(input_name);
        output_operation = self.graph.get_operation_by_name(output_name);
        
        t = self.read_tensor_from_image(image)
        
        
        with tf.Session(graph=self.graph) as sess:

            results = sess.run(output_operation.outputs[0],
                              {input_operation.outputs[0]: t})
        results = np.squeeze(results)

        top_k = results.argsort()[:][::-1]

        matched_light = top_k[0]

        if matched_light == 0:
            return TrafficLight.RED
        if matched_light == 1:
            return TrafficLight.YELLOW
        if matched_light == 2:
            return TrafficLight.GREEN 
        if matched_light == 3:
            return TrafficLight.UNKNOWN
        
#if __name__ == "__main__":
 #   get_classification()
