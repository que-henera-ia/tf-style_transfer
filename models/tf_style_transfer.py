import os
import tensorflow as tf
# Load compressed models from tensorflow_hub
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'

import numpy as np
import time
import functools

from tf_utils.manage_data import *
from tf_utils.process_images import *
from models.vgg import vgg19_layers

####

# !pip install git+https://github.com/tensorflow/docs
import tensorflow_docs.vis.embed as embed
import gc
####




###################################################################################################
#                                           MODEL FUNCTIONS                                       #
###################################################################################################


def gram_matrix(input_tensor):
  result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
  input_shape = tf.shape(input_tensor)
  num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
  return result/(num_locations)



class StyleContentModel(tf.keras.models.Model):
  def __init__(self, style_layers, content_layers,style_weight=1e-2,content_weight=1e4,learning_rate=0.02, beta_1=0.99, epsilon=1e-1):
    super(StyleContentModel, self).__init__()
    self.vgg = vgg19_layers(style_layers + content_layers)
    self.style_layers = style_layers
    self.content_layers = content_layers
    self.num_style_layers = len(style_layers)
    self.num_content_layers = len(content_layers)
    self.vgg.trainable = False
    self.style_weight=style_weight
    self.content_weight=content_weight

    self.opt = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1, epsilon=epsilon)

  def __call__(self, inputs):
    "Expects float input in [0,1]"
    inputs = inputs*255.0
    preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
    outputs = self.vgg(preprocessed_input)
    style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                      outputs[self.num_style_layers:])

    style_outputs = [gram_matrix(style_output)
                     for style_output in style_outputs]

    content_dict = {content_name: value
                    for content_name, value
                    in zip(self.content_layers, content_outputs)}

    style_dict = {style_name: value
                  for style_name, value
                  in zip(self.style_layers, style_outputs)}

    return {'content': content_dict, 'style': style_dict}

  def style_content_loss(self, outputs):
      style_outputs = outputs['style']
      content_outputs = outputs['content']
      style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-self.style_targets[name])**2) 
                            for name in style_outputs.keys()])
      style_loss *= self.style_weight / self.num_style_layers

      content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-self.content_targets[name])**2) 
                              for name in content_outputs.keys()])
      content_loss *= self.content_weight / self.num_content_layers
      loss = style_loss + content_loss
      return loss

  '''
  @tf.function is a decorator provided by TensorFlow that transforms Python functions into graph operations. This transformation enables 
  TensorFlow to compile and optimize the function’s computation, leading to enhanced performance and efficiency. Unlike traditional Python 
  functions, tf.function utilizes graph-based execution, which can significantly improve execution speed, especially for repetitive tasks.
  '''
  @tf.function()
  def train_step(self, image):
    with tf.GradientTape() as tape:
      outputs = self.__call__(image)
      loss = self.style_content_loss(outputs)

    grad = tape.gradient(loss, image)
    self.opt.apply_gradients([(grad, image)])
    image.assign(clip_0_1(image))
####################################################################################################



