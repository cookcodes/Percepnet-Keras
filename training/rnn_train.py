
from __future__ import print_function
from typing import Sequence

#import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import SimpleRNN
from keras.layers import Dropout
from keras.layers import concatenate
from keras.layers import Conv1D
from keras import initializers
from keras import losses
from keras import regularizers
from keras.constraints import min_max_norm
import h5py

from keras.constraints import Constraint
from keras import backend as K
from keras.callbacks import ModelCheckpoint
import numpy as np
import random

#import tensorflow as tf
#from keras.backend.tensorflow_backend import set_session
#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.42
#set_session(tf.Session(config=config))

def my_crossentropy(y_true, y_pred):
    return K.mean(2*K.abs(y_true-0.5) * K.binary_crossentropy(y_pred, y_true), axis=-1)

def mymask(y_true):
    return K.minimum(y_true+1., 1.)

def msse(y_true, y_pred):
    return K.mean(mymask(y_true) * K.square(K.sqrt(y_pred) - K.sqrt(y_true)), axis=-1)

def mycost(y_true, y_pred):
    return K.mean(mymask(y_true) * (10*K.square(K.square(K.sqrt(y_pred) - K.sqrt(y_true))) + K.square(K.sqrt(y_pred) - K.sqrt(y_true)) + 0.01*K.binary_crossentropy(y_pred, y_true)), axis=-1)
    
def gbcost(y_true, y_pred):
    return K.mean(mymask(y_true) * (10*K.square(K.square(K.sqrt(y_pred) - K.sqrt(y_true))) + K.square(K.sqrt(y_pred) - K.sqrt(y_true)) + 0.01*K.binary_crossentropy(y_pred, y_true)), axis=-1)
    
def rbcost(y_true, y_pred):
    return K.mean(mymask(y_true) * (K.square(K.sqrt(1-y_pred) - K.sqrt(1-y_true)) + 0.01*K.binary_crossentropy(1-y_pred, 1-y_true)), axis=-1)


def my_accuracy(y_true, y_pred):
    return K.mean(2*K.abs(y_true-0.5) * K.equal(y_true, K.round(y_pred)), axis=-1)

class WeightClip(Constraint):
    '''Clips the weights incident to each hidden unit to be inside a range
    '''
    def __init__(self, c=2):
        self.c = c

    def __call__(self, p):
        return K.clip(p, -self.c, self.c)

    def get_config(self):
        return {'name': self.__class__.__name__,
            'c': self.c}

mean= [1.49977617e+02, 2.58416061e+02, 3.24114287e+02, 3.36499261e+02,
 3.77253359e+02, 3.78842082e+02, 3.57965860e+02, 3.50937699e+02,
 3.30336109e+02, 2.85964070e+02, 2.55139451e+02, 2.28925847e+02,
 2.05563985e+02, 1.88992212e+02, 1.73476927e+02, 1.63679944e+02,
 1.46348951e+02, 1.29901599e+02, 1.21258336e+02, 1.13507282e+02,
 1.04923790e+02, 9.13304787e+01, 7.73656518e+01, 6.84674221e+01,
 6.28571237e+01, 5.98183763e+01, 4.97560230e+01, 1.66795795e+01,
 1.17776715e-01, 1.04928949e-01, 1.04522514e-01, 1.06600169e-01,
 1.09998411e-01, 1.16328487e-01, 7.79795789e-01, 7.48906024e-01,
 7.63785508e-01, 7.90205698e-01, 7.83844074e-01, 7.60185728e-01,
 7.28363072e-01, 7.22938180e-01, 7.19589773e-01, 7.04488197e-01,
 7.04837667e-01, 6.89970197e-01, 6.65663096e-01, 6.50594180e-01,
 6.53865545e-01, 6.58959487e-01, 6.42061315e-01, 6.37251674e-01,
 6.41502832e-01, 6.32374715e-01, 6.31881485e-01, 6.19003357e-01,
 6.03858191e-01, 5.85374468e-01, 5.71279354e-01, 5.63767848e-01,
 5.53812139e-01, 4.95578818e-01, 1.57462553e-01, 1.45038220e-01,
 1.45817444e-01, 1.50226251e-01, 1.56343914e-01, 1.64430522e-01,
 5.99644256e-01, 5.07925439e+08]
std= [9.48943362e+02, 1.22538053e+03, 1.27336286e+03, 1.15092918e+03,
 1.14645128e+03, 1.20168096e+03, 1.16712689e+03, 1.18759620e+03,
 1.20124740e+03, 1.11188198e+03, 1.07592528e+03, 1.02410738e+03,
 9.76602253e+02, 8.72284998e+02, 7.89747788e+02, 6.99113546e+02,
 6.34493270e+02, 6.12061284e+02, 5.52425487e+02, 5.13629579e+02,
 4.68399026e+02, 3.89667332e+02, 3.43376059e+02, 3.18090500e+02,
 2.98424602e+02, 2.83666340e+02, 2.38881812e+02, 8.39829965e+01,
 1.57760040e+00, 1.45105098e+00, 1.36318067e+00, 1.28701033e+00,
 1.22626335e+00, 1.20867922e+00, 2.85279751e-01, 2.87822393e-01,
 2.86545913e-01, 2.78652728e-01, 2.82478349e-01, 2.91316270e-01,
 3.00092113e-01, 3.00638986e-01, 2.96847897e-01, 2.89826314e-01,
 2.83058489e-01, 2.82778362e-01, 2.84345090e-01, 2.86813970e-01,
 2.72945143e-01, 2.62602871e-01, 2.62337358e-01, 2.55355726e-01,
 2.45500440e-01, 2.43858506e-01, 2.35832657e-01, 2.32271178e-01,
 2.26499394e-01, 2.22434765e-01, 2.19045730e-01, 2.14819696e-01,
 2.20300237e-01, 2.65284300e-01, 2.09639798e-01, 2.01054519e-01,
 2.01247058e-01, 2.03736700e-01, 2.07048809e-01, 2.11311224e-01,
 3.03199771e-01, 7.48641235e+09]

class CustomDataGen(tf.keras.utils.Sequence):
    
    def __init__(self, lenth,
                 batch_size,
                 data_index_begin,
                 data_index_end,
                 window_size,
                 ):
        
        self.len = lenth
        self.batch_size = batch_size
        self.data_index_begin = data_index_begin
        self.data_index_end = data_index_end
        self.window_size = window_size
        self.half = 0
        self.total_len = self.data_index_end - self.data_index_begin
        self.all_data = 0
        self.batch_count = 0
        with h5py.File('training.h5', 'r') as hf:
            self.all_data = hf['data'][self.data_index_begin:self.data_index_end]
            data_len = len(self.all_data)
            sequence = data_len//self.window_size
            self.all_data = np.reshape(self.all_data,(sequence,self.window_size,138))
            self.all_data[:,:,0:70] -= mean
            self.all_data[:,:,0:70] /= std
                    
    def on_epoch_end(self):
        self.half = random.randint(0,1)
        self.batch_count = 0
        np.random.shuffle(self.all_data)
            
    def __getitem__(self, index):
        return self.all_data[index*self.batch_size:(index+1)*self.batch_size,:,:70], [self.all_data[index*self.batch_size:(index+1)*self.batch_size,:,104:138], self.all_data[index*self.batch_size:(index+1)*self.batch_size,:,70:104]]
    
    def __len__(self):
        return self.len // self.batch_size

reg = 0.000001
constraint = WeightClip(0.499)

print('Build model...')
initializer = initializers.RandomUniform(minval=-0.1, maxval=0.1)

main_input = Input(shape=(None, 70), name='main_input')
tmp = Dense(128, activation='tanh', name='input_dense', kernel_constraint=constraint, bias_constraint=constraint, kernel_initializer=initializer)(main_input)

# conv 1X5  512
cnn1 = Conv1D(512, kernel_size=5, strides=1,use_bias=False, padding='causal',name='conv_layer1',kernel_initializer=initializer )(tmp)
# conv 1X3  512
cnn2 = Conv1D(512, kernel_size=3, strides=1,use_bias=False, padding='causal',name='conv_layer2',kernel_initializer=initializer)(cnn1)

# gru  512
#gru1 = CuDNNGRU(512, activation='tanh', recurrent_activation='sigmoid', return_sequences=True, name='noise_gru1', kernel_regularizer=regularizers.l2(reg), recurrent_regularizer=regularizers.l2(reg), kernel_constraint=constraint, recurrent_constraint=constraint, bias_constraint=constraint, kernel_initializer=initializer)(cnn2)
gru1 = GRU(512, return_sequences=True, name='noise_gru1', kernel_regularizer=regularizers.l2(reg), recurrent_regularizer=regularizers.l2(reg), kernel_constraint=constraint, recurrent_constraint=constraint, bias_constraint=constraint, kernel_initializer=initializer)(cnn2)
# gru  512
gru2 = GRU(512, return_sequences=True, name='noise_gru2', kernel_regularizer=regularizers.l2(reg), recurrent_regularizer=regularizers.l2(reg), kernel_constraint=constraint, recurrent_constraint=constraint, bias_constraint=constraint, kernel_initializer=initializer)(gru1)
# gru  512
gru3 = GRU(512, return_sequences=True, name='noise_gru3', kernel_regularizer=regularizers.l2(reg), recurrent_regularizer=regularizers.l2(reg), kernel_constraint=constraint, recurrent_constraint=constraint, bias_constraint=constraint, kernel_initializer=initializer)(gru2)
# gru  512
gru4 = GRU(512, return_sequences=True, name='noise_gru4', kernel_regularizer=regularizers.l2(reg), recurrent_regularizer=regularizers.l2(reg), kernel_constraint=constraint, recurrent_constraint=constraint, bias_constraint=constraint, kernel_initializer=initializer)(gru3)

#gb
gb_input = keras.layers.concatenate([cnn2, gru1, gru2, gru3, gru4])
gb_output = Dense(34, activation='sigmoid', name='denoise_output', kernel_constraint=constraint, bias_constraint=constraint, kernel_initializer=initializer)(gb_input)

#rb
rb_input = keras.layers.concatenate([cnn2, gru3])
rb_input2 = GRU(128, return_sequences=True, name='denoise_gru', kernel_regularizer=regularizers.l2(reg), recurrent_regularizer=regularizers.l2(reg), kernel_constraint=constraint, recurrent_constraint=constraint, bias_constraint=constraint, kernel_initializer=initializer)(rb_input)
rb_output = Dense(34, activation='sigmoid', name='denoise_rb_output', kernel_constraint=constraint, bias_constraint=constraint, kernel_initializer=initializer)(rb_input2)

model = Model(inputs=main_input, outputs=[gb_output, rb_output])

model.summary()

model.compile(loss=[gbcost, rbcost],
            metrics=[msse],
            optimizer=keras.optimizers.Adam(lr=0.0001,clipnorm=0.1) , loss_weights=[10, 5])

batch_size = 32

window_size = 2000

traingen = CustomDataGen(35200,batch_size,0,35200*window_size,window_size)
valgen = CustomDataGen(4800,batch_size,35200*window_size,40000*window_size,window_size)

print('Train...')

checkpoint_cb = ModelCheckpoint(
        'weights.{epoch:03d}-{val_loss:.2f}.h5', save_weights_only=False, save_freq='epoch')

model.step_counter = 0

model.fit(traingen,
          validation_data=valgen,
          epochs=150,
          callbacks=[checkpoint_cb],
          )

model.save("weights.hdf5")

