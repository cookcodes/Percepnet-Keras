   
from __future__ import print_function

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

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

#data_in_len = 3000
data_in_len = 2237

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
    #return K.mean(mymask(y_true) * (10*K.square(K.square(K.sqrt(y_pred) - K.sqrt(y_true))) + K.square(K.sqrt(y_pred) - K.sqrt(y_true))), axis=-1)

def gbcost(y_true, y_pred):
    return K.mean(mymask(y_true) * (10*K.square(K.square(K.sqrt(y_pred) - K.sqrt(y_true))) + K.square(K.sqrt(y_pred) - K.sqrt(y_true)) + 0.01*K.binary_crossentropy(y_pred, y_true)), axis=-1)
    #return K.mean(mymask(y_true) * (10*K.square(K.square(K.sqrt(y_pred) - K.sqrt(y_true))) + K.square(K.sqrt(y_pred) - K.sqrt(y_true))), axis=-1)


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

class MyModel(keras.Model):
    step_counter = 0
    def train_step(self, data):
        print()
        print("----Start of step: %d" % (self.step_counter,))
        self.step_counter += 1

        inputs, targets = data
        trainable_vars = self.trainable_variables
        with tf.GradientTape() as tape2:
            with tf.GradientTape() as tape1:
                preds = self(inputs, training=True)  # Forward pass
                print("inputs:")
                print(inputs)
                print("preds:")
                print(preds)
                # Compute the loss value
                # (the loss function is configured in `compile()`)
                print("targets:")
                print(targets)
                loss = self.compiled_loss(targets, preds)
                print("loss:")
                print(loss)
            # Compute first-order gradients
            dl_dw = tape1.gradient(loss, trainable_vars)
        # Compute second-order gradients
        d2l_dw2 = tape2.gradient(dl_dw, trainable_vars)

        print("Max of dl_dw[0]: %.4f" % tf.reduce_max(dl_dw[0]))
        print("Min of dl_dw[0]: %.4f" % tf.reduce_min(dl_dw[0]))
        print("Mean of dl_dw[0]: %.4f" % tf.reduce_mean(dl_dw[0]))
        print("-")
        print("Max of d2l_dw2[0]: %.4f" % tf.reduce_max(d2l_dw2[0]))
        print("Min of d2l_dw2[0]: %.4f" % tf.reduce_min(d2l_dw2[0]))
        print("Mean of d2l_dw2[0]: %.4f" % tf.reduce_mean(d2l_dw2[0]))

        # Combine first-order and second-order gradients
        grads = [0.5 * w1 + 0.5 * w2 for (w1, w2) in zip(d2l_dw2, dl_dw)]

        # Update weights
        self.optimizer.apply_gradients(zip(grads, trainable_vars))

        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(targets, preds)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

reg = 0.000001
constraint = WeightClip(0.499)

print('Build model...')
initializer = initializers.RandomUniform(minval=-0.5, maxval=0.5)

main_input = Input(shape=(data_in_len, 70), name='main_input',batch_size =1)
#main_input = Input(shape=(2000, 70), name='main_input', batch_size=32)
#FC 128
tmp = Dense(128, activation='tanh', name='input_dense', kernel_constraint=constraint, bias_constraint=constraint, kernel_initializer=initializer)(main_input)
#tmp = tf.keras.layers.Reshape((-1, 128, 1))(tmp) #reshape for conv1d

#convin = cov_layer_in(tmp)
#convin = tf.keras.layers.Reshape((-1,7, 128))(convin)
# conv 1X5  512
cnn1 = Conv1D(512, kernel_size=5, strides=1,use_bias=False, padding='causal',name='conv_layer1',kernel_initializer=initializer )(tmp)
# conv 1X3  512
cnn2 = Conv1D(512, kernel_size=3, strides=1,use_bias=False, padding='causal',name='conv_layer2',kernel_initializer=initializer)(cnn1)
#cnn2 = tf.keras.layers.Reshape((2000,512))(cnn2) #reshape after conv1d

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
#model = MyModel(inputs=main_input, outputs=[gb_output, rb_output])

model.summary()

#keras.utils.plot_model(model, show_shapes=True)

#model.compile(loss=[gbcost, rbcost],
#              metrics=[msse],
#              optimizer='adam', loss_weights=[10, 5], run_eagerly=True)

model.compile(loss=[gbcost, rbcost],
            metrics=[msse],
            optimizer=keras.optimizers.Adam(lr=0.0001,clipnorm=0.1) , loss_weights=[10, 5])


#model.compile(loss=[gbcost, rbcost],
#              metrics=[msse],
#              optimizer='RMSprop', loss_weights=[10, 5])

model.load_weights("weights.hdf5")

rnn_in = np.fromfile("../examples/rnn_in.dat",dtype="f")
total_len = len(rnn_in)
lines = total_len//70
rnn_in = np.reshape(rnn_in,(1,total_len//70,70))
rnn_outg = np.fromfile("../examples/rnn_outg.dat",dtype="f")
rnn_outr = np.fromfile("../examples/rnn_outr.dat",dtype="f")

rnn_outg = np.reshape(rnn_outg,(lines,34))
rnn_outr = np.reshape(rnn_outr,(lines,34))

print("rnn_in.shape:",rnn_in.shape)

g,r = model.predict(rnn_in,batch_size =1, steps = total_len//70)

test_layer = model.get_layer("noise_gru1")
get_output = K.function([model.layers[0].input], 
                        [test_layer.output, model.layers[-1].output])
[test_outputs, gr] = get_output(rnn_in)


output_dims = 512

#print("g.shape:",g.shape)
#print("r.shape:",r.shape)
print("test_outputs.shape:",test_outputs.shape)

#print(g)
#print(r)
#print(test_outputs)
test_outputs = np.reshape(test_outputs,(data_in_len,output_dims))

#test_layer_out = np.fromfile("../examples/dense_out1804289383.dat",dtype="f")
#test_layer_out = np.reshape(test_layer_out,(data_in_len,128))

#test_conv1_out = np.fromfile("../examples/conv1_out846930886.dat",dtype="f")
#test_conv1_out = np.reshape(test_conv1_out,(data_in_len,512))


test_conv1_out = np.fromfile("../examples/gru1_out1714636915.dat",dtype="f")
test_conv1_out = np.reshape(test_conv1_out,(data_in_len,output_dims))


#print(test_layer_out)
colslists = []

for i in range(data_in_len):
    result = np.allclose(test_outputs[i],test_conv1_out[i], rtol=1e-03, atol=1e-03)
    if not result:
        for j in range(output_dims):
            if abs(test_outputs[i][j] - test_conv1_out[i][j]) > 0.0001 :
                print(i," ",j," p:",test_outputs[i][j]," c:",test_conv1_out[i][j])
                colslists.insert(0,j)

result = np.allclose(test_outputs,test_conv1_out, rtol=1e-02, atol=1e-02)

print("compare result:",result)

colslists.sort()

print("colslists:",colslists)

g = np.reshape(g,(lines, 34))
r = np.reshape(r,(lines, 34))

print("begin to compare g, r")

for i in range(lines):
    result = np.allclose(g[i],rnn_outg[i], rtol=1e-03, atol=1e-03)
    if not result:
        print(i," g_python: ",g[i],"   g_c:  ",rnn_outg[i])
    result = np.allclose(r[i],rnn_outr[i], rtol=1e-03, atol=1e-03)
    if not result:
        print(i," r_python: ",r[i],"   r_c: ",rnn_outr[i])

#   print(i," gpython: ",g[i])
#   print(i,"     g_c: ",rnn_outg[i])
#   print(i," rpython: ",r[i])
#   print(i,"     r_c: ",rnn_outr[i])


g.astype('f').tofile("g_ref.dat")
r.astype('f').tofile("r_ref.dat")



