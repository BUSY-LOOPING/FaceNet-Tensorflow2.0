import tensorflow as tf
from keras.layers import Conv2D, MaxPool2D, Concatenate

class InceptionV1(tf.keras.layers.Layer):
    def __init__(self, filters_1x1, filters_3x3, filters_3x3_reduction, filters_5x5, filters_5x5_reduction, filters_pool_reduction, **kwargs):
        super(InceptionV1, self).__init__(**kwargs)
        
        # Define the 1x1 convolution branch
        self.branch1x1 = Conv2D(filters_1x1, (1, 1), padding='same', activation='relu')
        
        # Define the 3x3 convolution and reduction branch
        self.branch3x3 = Conv2D(filters_3x3, (3, 3), padding='same', activation='relu')
        self.branch3x3_reduction = Conv2D(filters_3x3_reduction, (1, 1), padding='same', activation='relu')
        
        # Define the 5x5 convolution and reduction branch
        self.branch5x5 = Conv2D(filters_5x5, (5, 5), padding='same', activation='relu')
        self.branch5x5_reduction = Conv2D(filters_5x5_reduction, (1, 1), padding='same', activation='relu')


        # Define the pooling and reduction branch
        self.branch_pool = MaxPool2D((3, 3), strides=(1, 1), padding='same')
        self.branch_pool_reduction = Conv2D(filters_pool_reduction, (1, 1), padding='same', activation='relu')


    def call(self, inputs):
        branch1x1 = self.branch1x1(inputs)
        
        branch3x3_reduction = self.branch3x3_reduction(inputs)
        branch3x3 = self.branch3x3(branch3x3_reduction)
        
        branch5x5_reduction = self.branch5x5_reduction(inputs)
        branch5x5 = self.branch5x5(branch5x5_reduction)
        
        branch_pool = self.branch_pool(inputs)
        branch_pool_reduction = self.branch_pool_reduction(branch_pool)
        
        # Concatenate the outputs of all branches
        output = Concatenate(axis=-1)([branch1x1, branch3x3, branch5x5, branch_pool_reduction])
        return output