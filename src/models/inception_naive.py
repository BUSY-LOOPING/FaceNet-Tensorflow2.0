import tensorflow as tf
from keras.layers import Conv2D, MaxPool2D, Concatenate

class InceptionNaive(tf.keras.layers.Layer):
    def __init__(self, filters_1x1, filters_3x3, filters_5x5, **kwargs):
        super(InceptionNaive, self).__init__(**kwargs)
        
        # Define the 1x1 convolution branch
        self.branch1x1 = Conv2D(filters_1x1, (1, 1), padding='same', activation='relu')
        
        # Define the 3x3 convolution branch
        self.branch3x3 = Conv2D(filters_3x3, (3, 3), padding='same', activation='relu')
        
        # Define the 5x5 convolution branch
        self.branch5x5 = Conv2D(filters_5x5, (5, 5), padding='same', activation='relu')
        
        # Define the pooling branch
        self.branch_pool = MaxPool2D((3, 3), strides=(1, 1), padding='same')


    def call(self, inputs):
        branch1x1 = self.branch1x1(inputs)
        
        branch3x3 = self.branch3x3(inputs)
        
        branch5x5 = self.branch5x5(inputs)
        
        branch_pool = self.branch_pool(inputs)
        
        # Concatenate the outputs of all branches
        output = Concatenate(axis=-1)([branch1x1, branch3x3, branch5x5, branch_pool])
        return output