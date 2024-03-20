from models.inception_naive import *
from models.inception_v1 import *

from keras.layers import Input
from keras.models import Model

i = Input(shape=(220, 220, 3))
inception_naive = InceptionNaive(120, 50, 30) (i)

model1 = Model(i, inception_naive, name = 'NaiveInception')
print(model1.summary())

naive_v1 = InceptionV1(
    filters_1x1 = 128,
    filters_3x3 = 256,
    filters_3x3_reduction = 64,
    filters_5x5 = 128,
    filters_5x5_reduction = 32,
    filters_pool_reduction = 64) (i)

model2 = Model(i, inception_naive, name = 'InceptionV1')
print(model2.summary())