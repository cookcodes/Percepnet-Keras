#ifndef RNN_DATA_H
#define RNN_DATA_H

#include "rnn.h"

struct RNNModel {
  int input_dense_size;
  const DenseLayer *input_dense;

  int conv_layer1_size;
  const ConvLayer *conv_layer1;

  int conv_layer2_size;
  const ConvLayer *conv_layer2;
      
  int noise_gru1_size;
  const GRULayer *noise_gru1;

  int noise_gru2_size;
  const GRULayer *noise_gru2;

  int noise_gru3_size;
  const GRULayer *noise_gru3;

  int noise_gru4_size;
  const GRULayer *noise_gru4;

  int denoise_gru_size;
  const GRULayer *denoise_gru;

  int denoise_output_size;
  const DenseLayer *denoise_output;

  int denoise_rb_output_size;
  const DenseLayer *denoise_rb_output;
};

struct RNNState {
  const RNNModel *model;
  float *noise_gru1_state;
  float *noise_gru2_state;
  float *noise_gru3_state;
  float *noise_gru4_state;
  float *denoise_gru_state;
  float * stored_cov_l1_in;
  float * stored_cov_l2_in;
  FILE * layer_out[8];
};


#endif
