/* Copyright (c) 2008-2011 Octasic Inc.
                 2012-2017 Jean-Marc Valin */
/*
   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

   - Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE FOUNDATION OR
   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <math.h>
#include "opus_types.h"
#include "common.h"
#include "arch.h"
#include "tansig_table.h"
#include "rnn.h"
#include "rnn_data.h"
#include <stdio.h>

static OPUS_INLINE float tansig_approx(float x)
{
    int i;
    float y, dy;
    float sign=1;
    /* Tests are reversed to catch NaNs */
    if(isnan(x))
    {
       printf("tansig_approx: nan detected");
    }

    if (!(x<8))
        return 1;
    if (!(x>-8))
        return -1;
#ifndef FIXED_POINT
    /* Another check in case of -ffast-math */
    if (celt_isnan(x))
       return 0;
#endif
    if (x<0)
    {
       x=-x;
       sign=-1;
    }
    i = (int)floor(.5f+25*x);
    x -= .04f*i;
    y = tansig_table[i];
    dy = 1-y*y;
    y = y + x*dy*(1 - y*x);
    return sign*y;
}

static OPUS_INLINE float sigmoid_approx(float x)
{
   return .5 + .5*tansig_approx(.5*x);
}

static OPUS_INLINE float relu(float x)
{
   return x < 0 ? 0 : x;
}


int compute_conv1D(const ConvLayer *layer, float *output, const float *input, int input_col){
   int i, j, k, m;
   int N, M, K;
   K = layer->kernelsize;
   M = layer->nb_inputs;
   N = layer->nb_outputs;
   
   // calculate output cols.
   int output_col = input_col-K+1;

   for (i=0;i<output_col;i++)
   {
      for (j=0;j<N;j++)
      {
         float sum = 0;
         for(k=0; k<K; k++)
         {
            for(m=0; m<M; m++){
               sum += layer->input_weights[(k*M + m )* N + j]*input[(i+k)*M+m];
            }
         }
         output[i*N+j] = WEIGHTS_SCALE*sum;
      }
   }
   return output_col;
}

void compute_dense(const DenseLayer *layer, float *output, const float *input)
{
   int i, j;
   int N, M;
   int stride;
   M = layer->nb_inputs;
   N = layer->nb_neurons;
   stride = N;
   for (i=0;i<N;i++)
   {
      /* Compute update gate. */
      float sum = layer->bias[i];
      for (j=0;j<M;j++)
         sum += layer->input_weights[j*stride + i]*input[j];
      output[i] = WEIGHTS_SCALE*sum;
   }

   if (layer->activation == ACTIVATION_SIGMOID) {
      for (i=0;i<N;i++)
         output[i] = sigmoid_approx(output[i]);
   } else if (layer->activation == ACTIVATION_TANH) {
      for (i=0;i<N;i++)
         output[i] = tansig_approx(output[i]);
   } else if (layer->activation == ACTIVATION_RELU) {
      for (i=0;i<N;i++)
         output[i] = relu(output[i]);
   } else {
     *(int*)0=0;
   }
}

void compute_gru(const GRULayer *gru, float *state, const float *input)
{
   int i, j;
   int N, M;
   int stride;
   float z[MAX_NEURONS];
   float r[MAX_NEURONS];
   float h[MAX_NEURONS];
   M = gru->nb_inputs;
   N = gru->nb_neurons;
   stride = 3*N;
   for (i=0;i<N;i++)
   {
      /* Compute update gate. */
      float sum = gru->bias[i]+gru->bias[stride+i];
      for (j=0;j<M;j++)
         sum += gru->input_weights[j*stride + i]*input[j];
      for (j=0;j<N;j++) 
         sum += gru->recurrent_weights[j*stride + i]*state[j];
      z[i] = sigmoid_approx(WEIGHTS_SCALE*sum);
   }
   for (i=0;i<N;i++)
   {
      /* Compute reset gate. */
      float sum = gru->bias[N + i]+gru->bias[stride+N+i];
      for (j=0;j<M;j++)
         sum += gru->input_weights[N + j*stride + i]*input[j];
      for (j=0;j<N;j++)
         sum += gru->recurrent_weights[N + j*stride + i]*state[j];
      r[i] = sigmoid_approx(WEIGHTS_SCALE*sum);
   }
   for (i=0;i<N;i++)
   {
      /* Compute output. */
      float sum = gru->bias[2*N + i];
      for (j=0;j<M;j++)
         sum += gru->input_weights[2*N + j*stride + i]*input[j];
      //for (j=0;j<N;j++)
      //   sum += (gru->recurrent_weights[2*N + j*stride + i]*state[j])*r[j];
      float sum_r = gru->bias[stride + 2*N + i];
      for (j=0;j<N;j++){
         sum_r += (gru->recurrent_weights[2*N + j*stride + i]*state[j]);
      }
      sum += sum_r * r[i];
      
      if (gru->activation == ACTIVATION_SIGMOID) sum = sigmoid_approx(WEIGHTS_SCALE*sum);
      else if (gru->activation == ACTIVATION_TANH) sum = tansig_approx(WEIGHTS_SCALE*sum);
      else if (gru->activation == ACTIVATION_RELU) sum = relu(WEIGHTS_SCALE*sum);
      else *(int*)0=0;
      h[i] = z[i]*state[i] + (1-z[i])*sum;
   }
   for (i=0;i<N;i++)
      state[i] = h[i];
}


void compute_rnn(RNNState *rnn, float *gains, float *rb_gains, const float *input) {
  int i;
  float dense_out[MAX_NEURONS];

  float conv1_out[MAX_NEURONS];
  float conv2_out[MAX_NEURONS];

  float gb_input[MAX_NEURONS*5];//512*5 = 2560

  float denoise_gru_input[MAX_NEURONS*2];//512*2 = 1024
  
  compute_dense(rnn->model->input_dense, dense_out, input);
  
  memmove(rnn->stored_cov_l1_in, &rnn->stored_cov_l1_in[128],4*128*sizeof(float));
  memcpy(&rnn->stored_cov_l1_in[4*128], dense_out,128*sizeof(float));

  //compute two convs
  compute_conv1D(rnn->model->conv_layer1, conv1_out, rnn->stored_cov_l1_in,5);
  
  memmove(rnn->stored_cov_l2_in, &rnn->stored_cov_l2_in[512],2*512*sizeof(float));
  memcpy(&rnn->stored_cov_l2_in[2*512], conv1_out,512*sizeof(float));

  compute_conv1D(rnn->model->conv_layer2, conv2_out, rnn->stored_cov_l2_in,3);
  
  //compute four grus
  compute_gru(rnn->model->noise_gru1, rnn->noise_gru1_state,conv2_out);
  
  compute_gru(rnn->model->noise_gru2, rnn->noise_gru2_state,rnn->noise_gru1_state);
  
  compute_gru(rnn->model->noise_gru3, rnn->noise_gru3_state,rnn->noise_gru2_state); 
  
  compute_gru(rnn->model->noise_gru4, rnn->noise_gru4_state,rnn->noise_gru3_state);
  
  //concate for gb dense
  for (i=0;i<MAX_NEURONS;i++) gb_input[i] = conv2_out[i];
  for (i=0;i<MAX_NEURONS;i++) gb_input[i+MAX_NEURONS] = rnn->noise_gru1_state[i];
  for (i=0;i<MAX_NEURONS;i++) gb_input[i+2*MAX_NEURONS] = rnn->noise_gru2_state[i];
  for (i=0;i<MAX_NEURONS;i++) gb_input[i+3*MAX_NEURONS] = rnn->noise_gru3_state[i];
  for (i=0;i<MAX_NEURONS;i++) gb_input[i+4*MAX_NEURONS] = rnn->noise_gru4_state[i];

  //coumpute gb
  compute_dense(rnn->model->denoise_output, gains, gb_input);

  //concate for rb gru
  for (i=0;i<MAX_NEURONS;i++) denoise_gru_input[i] = conv2_out[i];
  for (i=0;i<MAX_NEURONS;i++) denoise_gru_input[i+MAX_NEURONS] = rnn->noise_gru3_state[i];
  
  compute_gru(rnn->model->denoise_gru, rnn->denoise_gru_state,denoise_gru_input);
  
  compute_dense(rnn->model->denoise_rb_output, rb_gains, rnn->denoise_gru_state);

}
