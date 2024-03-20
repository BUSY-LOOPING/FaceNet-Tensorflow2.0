import tensorflow as tf
import numpy as np

def load_image(path) :
    byte_img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(byte_img)
    return img