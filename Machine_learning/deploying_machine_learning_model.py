from __future__ import absolute_import, division, print_function, unicode_literals

import pathlib
import pprint
import os
import matplotlib.pyplot as plt
import pandas as pd
#import seaborn as sns
import numpy as np

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow import keras

new_model = tf.keras.models.load_model('DropLev_volt_vol_st_prediction_model.h5')

new_model.summary()

csvfile=str('pyridine_alldata.csv')
input_array=pd.read_csv(csvfile, delimiter='\t', header=None)
input_array=input_array.drop(input_array.columns[908:912], axis=1)
input_array=input_array.drop(input_array.columns[904:907], axis=1)
input_array=input_array.drop(input_array.columns[902], axis=1)

input_array=np.asarray(input_array)
prediction=new_model.predict(input_array)





