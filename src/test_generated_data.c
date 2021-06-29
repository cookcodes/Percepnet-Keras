/* Copyright (c) 2018 Gregor Richards
 * Copyright (c) 2017 Mozilla */
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

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "kiss_fft.h"
#include "common.h"
#include <math.h>
#include "rnnoise.h"
#include "pitch.h"
#include "arch.h"
#include "rnn.h"
#include "rnn_data.h"
#include "erbband_filter.h"

#define FRAME_SIZE_SHIFT 2
#define FRAME_SIZE (120<<FRAME_SIZE_SHIFT)
#define WINDOW_SIZE (2*FRAME_SIZE)
#define FREQ_SIZE (FRAME_SIZE + 1)

#define COMB_M 3

#define PITCH_MIN_PERIOD 60
#define PITCH_MAX_PERIOD 768
#define PITCH_FRAME_SIZE 960
#define PITCH_BUF_SIZE (PITCH_MAX_PERIOD+PITCH_FRAME_SIZE)

#define FRAME_LOOKAHEAD 5 //9(PITCH_MAX_PERIOD*COMB_M/WINDOW_SIZE + 0.5)
#define FRAME_LOOKAHEAD_SIZE (FRAME_LOOKAHEAD*FRAME_SIZE)
#define COMB_BUF_SIZE (FRAME_LOOKAHEAD*2*FRAME_SIZE+PITCH_FRAME_SIZE)

#define SQUARE(x) ((x)*(x))

#define NB_BANDS 34

#define CEPS_MEM 8
#define NB_DELTA_CEPS 6

#define NB_FEATURES (NB_BANDS+3*NB_DELTA_CEPS+2)


#ifndef TRAINING
#define TRAINING 0
#endif

#define SPEECH_FILE_TOTAL 14000

/* The built-in model, used if no file is given as input */
extern const struct RNNModel rnnoise_model_orig;

//static const opus_int16 eband5ms[] = {
/*0  200 400 600 800  1k 1.2 1.4 1.6  2k 2.4 2.8 3.2  4k 4.8 5.6 6.8  8k 9.6 12k 15.6 20k*/
//  0,  1,  2,  3,  4,  5,  6,  7,  8, 10, 12, 14, 16, 20, 24, 28, 34, 40, 48, 60, 78, 100
//};


typedef struct {
  int init;
  kiss_fft_state *kfft;
  float half_window[FRAME_SIZE];
  float dct_table[NB_BANDS*NB_BANDS];
  float comb_hann_window[COMB_M*2+1];
  float power_noise_attenuation;
  float n0;/*noise-masking-tone threshold*/
} CommonState;

struct DenoiseState {
  float analysis_mem[FRAME_SIZE];
  float cepstral_mem[CEPS_MEM][NB_BANDS];
  int memid;
  float synthesis_mem[FRAME_SIZE];
  float pitch_buf[PITCH_BUF_SIZE];
  float comb_buf[COMB_BUF_SIZE];  /* added for comb */
  float pitch_enh_buf[PITCH_BUF_SIZE];
  float last_gain;
  int last_period;
  float pitch_corr;  /* pitch correlation */
  float mem_hp_x[2];
  float lastg[NB_BANDS];
  RNNState rnn; 
};

void compute_band_energy(float *bandE, const kiss_fft_cpx *X) {
  int i;
  float sum[NB_BANDS] = {0};
  for (i=0;i<NB_BANDS;i++)
  {
    int j;
    //int band_size;
    //band_size = (eband5ms[i+1]-eband5ms[i])<<FRAME_SIZE_SHIFT;
    int low_nfft_idx = erbweights[i].begin;
    int high_nfft_idx = erbweights[i].end;
    for(j=low_nfft_idx; j<high_nfft_idx; j++){
      float tmp;
      tmp = SQUARE(X[j].r);
      tmp += SQUARE(X[j].i);
      sum[i] += tmp*erbweights[i].weights[j-low_nfft_idx];
    }
  }
  //sum[0] *= 2;
  //sum[NB_BANDS-1] *= 2;
  for (i=0;i<NB_BANDS;i++)
  {
    bandE[i] = sum[i];
  }
}

void compute_band_corr(float *bandE, const kiss_fft_cpx *X, const kiss_fft_cpx *P) {
  int i;
  float sum[NB_BANDS] = {0};
  for (i=0;i<NB_BANDS;i++)
  {
    int j;
    //int band_size;
    //band_size = (eband5ms[i+1]-eband5ms[i])<<FRAME_SIZE_SHIFT;
    int low_nfft_idx = erbweights[i].begin;
    int high_nfft_idx = erbweights[i].end;
    for(j=low_nfft_idx; j<high_nfft_idx; j++){
      float tmp;
      tmp = X[j].r * P[j].r;
      tmp += X[j].i * P[j].i;
      sum[i] += tmp*erbweights[i].weights[j-low_nfft_idx];
    }
   }

  //sum[0] *= 2;
  //sum[NB_BANDS-1] *= 2;
  for (i=0;i<NB_BANDS;i++)
  {
    bandE[i] = fabs(sum[i]);
  }
}

void interp_band_gain(float *g, const float *bandE) {
  int i;
  memset(g, 0, FREQ_SIZE);
  for (i=0;i<NB_BANDS-1;i++)
  {
    int j;
    //int band_size;
    //band_size = (eband5ms[i+1]-eband5ms[i])<<FRAME_SIZE_SHIFT;
    int low_nfft_idx = erbweights[i].begin;
    int high_nfft_idx = erbweights[i].end;
    for(j=low_nfft_idx; j<high_nfft_idx; j++){
      g[j] += erbweights[i].weights[j-low_nfft_idx]*bandE[i];
    }
  }
}

CommonState common;

static void check_init() {
  int i;
  float temp_sum=0;

  if (common.init) return;
  common.kfft = opus_fft_alloc_twiddles(2*FRAME_SIZE, NULL, NULL, NULL, 0);
  for (i=0;i<FRAME_SIZE;i++)
    common.half_window[i] = sin(.5*M_PI*sin(.5*M_PI*(i+.5)/FRAME_SIZE) * sin(.5*M_PI*(i+.5)/FRAME_SIZE));
  for (i=0;i<NB_BANDS;i++) {
    int j;
    for (j=0;j<NB_BANDS;j++) {
      common.dct_table[i*NB_BANDS + j] = cos((i+.5)*j*M_PI/NB_BANDS);
      if (j==0) common.dct_table[i*NB_BANDS + j] *= sqrt(.5);
    }
  }

  for (i=1;i<COMB_M*2+2; i++){
    common.comb_hann_window[i-1] = 0.5 - 0.5*cos(2.0*M_PI*i/(COMB_M*2+2));
    printf("comb_hann_window[%d]:%f\n", i-1, common.comb_hann_window[i-1]);
    temp_sum += common.comb_hann_window[i-1];
    printf("temp_sum:%f\n", temp_sum);
  }
  for (i=1;i<COMB_M*2+2; i++){
    common.comb_hann_window[i-1] /= temp_sum;
    printf("comb_hann_window[%d]:%f\n", i-1, common.comb_hann_window[i-1]);
  }
  common.power_noise_attenuation = 0;
  for (i=1;i<COMB_M*2+2; i++){
    common.power_noise_attenuation += common.comb_hann_window[i-1]*common.comb_hann_window[i-1];
  }
  printf("common.power_noise_attenuation:%f\n",common.power_noise_attenuation);
  common.n0 = 0.03;

  common.init = 1;
}

static void dct(float *out, const float *in) {
  int i;
  check_init();
  for (i=0;i<NB_BANDS;i++) {
    int j;
    float sum = 0;
    for (j=0;j<NB_BANDS;j++) {
      sum += in[j] * common.dct_table[j*NB_BANDS + i];
    }
    out[i] = sum*sqrt(2./22);
  }
}

#if 0
static void idct(float *out, const float *in) {
  int i;
  check_init();
  for (i=0;i<NB_BANDS;i++) {
    int j;
    float sum = 0;
    for (j=0;j<NB_BANDS;j++) {
      sum += in[j] * common.dct_table[i*NB_BANDS + j];
    }
    out[i] = sum*sqrt(2./22);
  }
}
#endif

static void forward_transform(kiss_fft_cpx *out, const float *in) {
  int i;
  kiss_fft_cpx x[WINDOW_SIZE];
  kiss_fft_cpx y[WINDOW_SIZE];
  check_init();
  for (i=0;i<WINDOW_SIZE;i++) {
    x[i].r = in[i];
    x[i].i = 0;
  }
  opus_fft(common.kfft, x, y, 0);
  for (i=0;i<FREQ_SIZE;i++) {
    out[i] = y[i];
  }
}

static void inverse_transform(float *out, const kiss_fft_cpx *in) {
  int i;
  kiss_fft_cpx x[WINDOW_SIZE];
  kiss_fft_cpx y[WINDOW_SIZE];
  check_init();
  for (i=0;i<FREQ_SIZE;i++) {
    x[i] = in[i];
  }
  for (;i<WINDOW_SIZE;i++) {
    x[i].r = x[WINDOW_SIZE - i].r;
    x[i].i = -x[WINDOW_SIZE - i].i;
  }
  opus_fft(common.kfft, x, y, 0);
  /* output in reverse order for IFFT. */
  out[0] = WINDOW_SIZE*y[0].r;
  for (i=1;i<WINDOW_SIZE;i++) {
    out[i] = WINDOW_SIZE*y[WINDOW_SIZE - i].r;
  }
}

static void apply_window(float *x) {
  int i;
  check_init();
  for (i=0;i<FRAME_SIZE;i++) {
    x[i] *= common.half_window[i];
    x[WINDOW_SIZE - 1 - i] *= common.half_window[i];
  }
}

int rnnoise_get_size() {
  return sizeof(DenoiseState);
}

int rnnoise_get_frame_size() {
  return FRAME_SIZE;
}

int rnnoise_init(DenoiseState *st, RNNModel *model) {
  memset(st, 0, sizeof(*st));
  /*
  if (model)
    st->rnn.model = model;
  else
    st->rnn.model = &rnnoise_model_orig;
  st->rnn.vad_gru_state = calloc(sizeof(float), st->rnn.model->vad_gru_size);
  st->rnn.noise_gru_state = calloc(sizeof(float), st->rnn.model->noise_gru_size);
  st->rnn.denoise_gru_state = calloc(sizeof(float), st->rnn.model->denoise_gru_size);
  */
  return 0;
}

DenoiseState *rnnoise_create(RNNModel *model) {
  DenoiseState *st;
  st = malloc(rnnoise_get_size());
  rnnoise_init(st, model);
  return st;
}

void rnnoise_destroy(DenoiseState *st) {
  //free(st->rnn.vad_gru_state);
  //free(st->rnn.noise_gru_state);
  //free(st->rnn.denoise_gru_state);
  free(st);
}

#if TRAINING
//int lowpass = FREQ_SIZE;
//int band_lp = NB_BANDS;
#endif

static void frame_analysis(DenoiseState *st, kiss_fft_cpx *X, float *Ex, const float *in) {
  int i;
  float x[WINDOW_SIZE];
  RNN_COPY(x, st->analysis_mem, FRAME_SIZE);
  for (i=0;i<FRAME_SIZE;i++) x[FRAME_SIZE + i] = in[i];
  RNN_COPY(st->analysis_mem, in, FRAME_SIZE);
  apply_window(x);
  forward_transform(X, x);
#if TRAINING
  //for (i=lowpass;i<FREQ_SIZE;i++)
  //  X[i].r = X[i].i = 0;
#endif
  compute_band_energy(Ex, X);
}

static int compute_frame_features(DenoiseState *st, kiss_fft_cpx *X, kiss_fft_cpx *P,
                                  float *Ex, float *Ep, float *Exp, float *features, const float *in) {
  int i,k;
  float E = 0;
  float *ceps_0, *ceps_1, *ceps_2;
  float spec_variability = 0;
  float Ly[NB_BANDS];
  float p[WINDOW_SIZE];
  float pitch_buf[PITCH_BUF_SIZE>>1];
  int pitch_index;
  float gain;
  float *(pre[1]);
  float tmp[NB_BANDS];
  float follow, logMax;
  float pitch_corr;

  //frame_analysis(st, X, Ex, in);
  RNN_MOVE(st->comb_buf, &st->comb_buf[FRAME_SIZE], COMB_BUF_SIZE-FRAME_SIZE);
  RNN_COPY(&st->comb_buf[COMB_BUF_SIZE-FRAME_SIZE], in, FRAME_SIZE);

  RNN_MOVE(st->pitch_buf, &st->pitch_buf[FRAME_SIZE], PITCH_BUF_SIZE-FRAME_SIZE);
  RNN_COPY(&st->pitch_buf[PITCH_BUF_SIZE-FRAME_SIZE], &st->comb_buf[COMB_BUF_SIZE-FRAME_SIZE*4], FRAME_SIZE);

  //float incombn[FRAME_SIZE];
  //RNN_COPY(incombn,&st->pitch_buf[PITCH_BUF_SIZE-FRAME_SIZE*4], FRAME_SIZE);

  frame_analysis(st, X, Ex, &st->comb_buf[COMB_BUF_SIZE-FRAME_SIZE*4]); 

  pre[0] = &st->pitch_buf[0];
  pitch_downsample(pre, pitch_buf, PITCH_BUF_SIZE, 1);
  pitch_search(pitch_buf+(PITCH_MAX_PERIOD>>1), pitch_buf, PITCH_FRAME_SIZE,
               PITCH_MAX_PERIOD-3*PITCH_MIN_PERIOD, &pitch_index, &pitch_corr);
  pitch_index = PITCH_MAX_PERIOD-pitch_index;

  gain = remove_doubling(pitch_buf, PITCH_MAX_PERIOD, PITCH_MIN_PERIOD,
          PITCH_FRAME_SIZE, &pitch_index, st->last_period, st->last_gain);
  st->last_period = pitch_index;
  st->last_gain = gain;
  st->pitch_corr = pitch_corr;

  //for (i=0;i<WINDOW_SIZE;i++)
  //  p[i] = st->pitch_buf[PITCH_BUF_SIZE-WINDOW_SIZE-pitch_index+i];

  for (i=0;i<WINDOW_SIZE;i++)
      p[i]=0;

  for (k=-COMB_M;k<COMB_M+1; k++){
    for (i=0;i<WINDOW_SIZE;i++)
      p[i] += st->comb_buf[COMB_BUF_SIZE-FRAME_SIZE*(COMB_M)-WINDOW_SIZE-pitch_index*k+i]*common.comb_hann_window[k+COMB_M];
  } 

  apply_window(p);
  forward_transform(P, p);
  compute_band_energy(Ep, P);
  compute_band_corr(Exp, X, P);
  for (i=0;i<NB_BANDS;i++) Exp[i] = Exp[i]/sqrt(.001+Ex[i]*Ep[i]);
  //dct(tmp, Exp);
  //for (i=0;i<NB_DELTA_CEPS;i++) features[NB_BANDS+2*NB_DELTA_CEPS+i] = tmp[i];
  //features[NB_BANDS+2*NB_DELTA_CEPS] -= 1.3;
  //features[NB_BANDS+2*NB_DELTA_CEPS+1] -= 0.9;
  //features[NB_BANDS+3*NB_DELTA_CEPS] = .01*(pitch_index-300);
  //logMax = -2;
  //follow = -2;
  for (i=0;i<NB_BANDS;i++) {
    //Ly[i] = log10(1e-2+Ex[i]);
    //Ly[i] = MAX16(logMax-7, MAX16(follow-1.5, Ly[i]));
    //logMax = MAX16(logMax, Ly[i]);
    //follow = MAX16(follow-1.5, Ly[i]);
    E += Ex[i];
  }
  if (!TRAINING && E < 0.04) {
    /* If there's no audio, avoid messing up the state. */
    RNN_CLEAR(features, NB_FEATURES);
    return 1;
  }

  /*
  dct(features, Ly);
  features[0] -= 12;
  features[1] -= 4;
  ceps_0 = st->cepstral_mem[st->memid];
  ceps_1 = (st->memid < 1) ? st->cepstral_mem[CEPS_MEM+st->memid-1] : st->cepstral_mem[st->memid-1];
  ceps_2 = (st->memid < 2) ? st->cepstral_mem[CEPS_MEM+st->memid-2] : st->cepstral_mem[st->memid-2];
  for (i=0;i<NB_BANDS;i++) ceps_0[i] = features[i];
  st->memid++;
  for (i=0;i<NB_DELTA_CEPS;i++) {
    features[i] = ceps_0[i] + ceps_1[i] + ceps_2[i];
    features[NB_BANDS+i] = ceps_0[i] - ceps_2[i];
    features[NB_BANDS+NB_DELTA_CEPS+i] =  ceps_0[i] - 2*ceps_1[i] + ceps_2[i];
  }
  /* Spectral variability features. 
  if (st->memid == CEPS_MEM) st->memid = 0;
  for (i=0;i<CEPS_MEM;i++)
  {
    int j;
    float mindist = 1e15f;
    for (j=0;j<CEPS_MEM;j++)
    {
      int k;
      float dist=0;
      for (k=0;k<NB_BANDS;k++)
      {
        float tmp;
        tmp = st->cepstral_mem[i][k] - st->cepstral_mem[j][k];
        dist += tmp*tmp;
      }
      if (j!=i)
        mindist = MIN32(mindist, dist);
    }
    spec_variability += mindist;
  }
  features[NB_BANDS+3*NB_DELTA_CEPS+1] = spec_variability/CEPS_MEM-2.1;
  */
  return TRAINING && E < 0.04;
}

static void frame_synthesis(DenoiseState *st, float *out, const kiss_fft_cpx *y) {
  float x[WINDOW_SIZE];
  int i;
  inverse_transform(x, y);
  apply_window(x);
  for (i=0;i<FRAME_SIZE;i++) out[i] = x[i] + st->synthesis_mem[i];
  RNN_COPY(st->synthesis_mem, &x[FRAME_SIZE], FRAME_SIZE);
}

static void biquad(float *y, float mem[2], const float *x, const float *b, const float *a, int N) {
  int i;
  for (i=0;i<N;i++) {
    float xi, yi;
    xi = x[i];
    yi = x[i] + mem[0];
    mem[0] = mem[1] + (b[0]*(double)xi - a[0]*(double)yi);
    mem[1] = (b[1]*(double)xi - a[1]*(double)yi);
    y[i] = yi;
  }
}

void pitch_filter(kiss_fft_cpx *X, const kiss_fft_cpx *P, const float *Ex, const float *Ep,
                  const float *Exp, const float *g, const float *r) {
  int i;
  //float r[NB_BANDS];
  float rf[FREQ_SIZE] = {0};
  /*
  for (i=0;i<NB_BANDS;i++) {
  
#if 0
    if (Exp[i]>g[i]) r[i] = 1;
    else r[i] = Exp[i]*(1-g[i])/(.001 + g[i]*(1-Exp[i]));
    r[i] = MIN16(1, MAX16(0, r[i]));
#else
    if (Exp[i]>g[i]) r[i] = 1;
    else r[i] = SQUARE(Exp[i])*(1-SQUARE(g[i]))/(.001 + SQUARE(g[i])*(1-SQUARE(Exp[i])));
    r[i] = sqrt(MIN16(1, MAX16(0, r[i])));
#endif
    r[i] *= sqrt(Ex[i]/(1e-8+Ep[i]));
  }
  */
  interp_band_gain(rf, r);
  for (i=0;i<FREQ_SIZE;i++) {
    //X[i].r += rf[i]*P[i].r;
    //X[i].i += rf[i]*P[i].i;
    X[i].r =(1-rf[i])*X[i].r + rf[i]*P[i].r;
    X[i].i =(1-rf[i])*X[i].i + rf[i]*P[i].i;

  }
  /*
  float newE[NB_BANDS];
  compute_band_energy(newE, X);
  float norm[NB_BANDS];
  float normf[FREQ_SIZE]={0};
  for (i=0;i<NB_BANDS;i++) {
    norm[i] = sqrt(Ex[i]/(1e-8+newE[i]));
  }
  interp_band_gain(normf, norm);
  for (i=0;i<FREQ_SIZE;i++) {
    X[i].r *= normf[i];
    X[i].i *= normf[i];
  }*/
}

float rnnoise_process_frame(DenoiseState *st, DenoiseState *st_current,float *out, const float *in) {
  int i;
  kiss_fft_cpx X[FREQ_SIZE];
  kiss_fft_cpx P[WINDOW_SIZE];
  float x[FRAME_SIZE];
  float Ex[NB_BANDS], Ep[NB_BANDS];
  float Exp[NB_BANDS];
  float features[NB_FEATURES];
  float g[NB_BANDS];
  float gf[FREQ_SIZE]={1};
  float vad_prob = 0;
  int silence;
  float r[NB_BANDS];

  static const float a_hp[2] = {-1.99599, 0.99600};
  static const float b_hp[2] = {-2, 1};
  biquad(x, st->mem_hp_x, in, b_hp, a_hp, FRAME_SIZE);
  silence = compute_frame_features(st, X, P, Ex, Ep, Exp, features, x);

  if (!silence) {
    //compute_rnn(&st->rnn, g, &vad_prob, features);
    pitch_filter(X, P, Ex, Ep, Exp, g, r);
    /*
    for (i=0;i<NB_BANDS;i++) {
      float alpha = .6f;
      g[i] = MAX16(g[i], alpha*st->lastg[i]);
      st->lastg[i] = g[i];
    }
    interp_band_gain(gf, g);
#if 1
    for (i=0;i<FREQ_SIZE;i++) {
      X[i].r *= gf[i];
      X[i].i *= gf[i];
    }
#endif
   */
  }

  frame_synthesis(st, out, X);
  return 0;
}

float rnnoise_process_frame_for_validate_data(DenoiseState *st, float *out, const float *in, float * g, float *r) {
  int i;
  kiss_fft_cpx X[FREQ_SIZE];
  kiss_fft_cpx P[WINDOW_SIZE];
  float x[FRAME_SIZE];
  float Ex[NB_BANDS], Ep[NB_BANDS];
  float Exp[NB_BANDS];
  float features[NB_FEATURES];
  float gf[FREQ_SIZE]={1};
  float vad_prob = 0;
  int silence;

  static const float a_hp[2] = {-1.99599, 0.99600};
  static const float b_hp[2] = {-2, 1};
  biquad(x, st->mem_hp_x, in, b_hp, a_hp, FRAME_SIZE);
  silence = compute_frame_features(st, X, P, Ex, Ep, Exp, features, x);


  //applied the gain here, envelop filter?

  if (!silence) {
    
    for (i=0;i<NB_BANDS;i++) {
      float alpha = .6f;
      g[i] = MAX16(g[i], alpha*st->lastg[i]);
      st->lastg[i] = g[i];
    }
    interp_band_gain(gf, g);
    for (i=0;i<FREQ_SIZE;i++) {
      X[i].r *= gf[i];
      X[i].i *= gf[i];
    }
   //applied pitch filter after gain filter

   pitch_filter(X, P, Ex, Ep, Exp, g, r);

  }

  frame_synthesis(st, out, X);
  return 0;
}

void estimate_phat_corr(CommonState st, float *Eyp, float *Ephatp){
  for(int i=0; i<NB_BANDS; i++){
    Ephatp[i] = Eyp[i]/sqrt((1-st.power_noise_attenuation)*Eyp[i]*Eyp[i] + st.power_noise_attenuation);
  }
}

void filter_strength_calc(float *Exp, float *Eyp, float *Ephatp, float* r){
  float alpha;
  float a;
  float b;
  for(int i=0; i<NB_BANDS; ++i){
    if(isnan(Ephatp[i])){printf("Ephatp[%d] is NAN, Eyp:%f\n",i, Eyp[i]);}
    //if(isnan(Exp[i])){printf("Exp[%d] is NAN",i);}
    //if(isnan(Eyp[i])){printf("Eyp[%d] is NAN",i);}
    a = Ephatp[i]*Ephatp[i] - Exp[i]*Exp[i];
    b = Ephatp[i]*Eyp[i]*(1-Exp[i]*Exp[i]);

    alpha = (sqrt(MAX16(0, b*b + a *(Exp[i]*Exp[i]-Eyp[i]*Eyp[i])))-b)/(a+1e-8);
    if(isnan(alpha)){printf("alpha[%d] is NAN, a= %f, \n",i, a);}
    r[i] = alpha/(1+alpha);
    if(isnan(r[i])){printf("r[%d] is NAN \n",i);}
    printf("a:%f  b:%f, alpha: %f, r:%f\n", a,b, alpha, r[i]);
  }
}

void calc_ideal_gain(float *X, float *Y, float* g){
  for(int i=0; i<NB_BANDS; ++i){
    g[i] = X[i]/(.0001+Y[i]);
  }
}

void adjust_gain_strength_by_condition(CommonState st, float *Ephatp, float * Eyp, float *Exp, float* g, float* r){
  float g_att;
  for(int i=0; i<NB_BANDS; ++i){
    //if(isnan(r[i])){printf("r[%d] is NAN",i);}
    if(isnan(g[i])){printf("g[%d] is NAN",i);}
    if(Ephatp[i]<Exp[i]){  //raoxy modify add abs to avoid Ephatp is negative number
      g_att = sqrt((1+st.n0-Exp[i]*Exp[i])/(1+st.n0-Ephatp[i]*Ephatp[i]));
      r[i] = 1;
      g[i] *= g_att;
      printf("g_att:%f\n", g_att);
    }
    printf("g:%f r:%f\n", g[i], r[i]);
    if(r[i]>1)printf("r>1 detected\n", r[i]);
    if(r[i]<0){
      printf("r<0 detected, r[i]:%f, Ephatp[i]: %f, Eyp[i]:%f, Exp[i]:%f \n", r[i],Ephatp[i],Eyp[i],Exp[i]);
      r[i]=0;
    }
  }
}

#if TRAINING

static float uni_rand() {
  return rand()/(double)RAND_MAX-.5;
}

static void rand_resp(float *a, float *b) {
  a[0] = .75*uni_rand();
  a[1] = .75*uni_rand();
  b[0] = .75*uni_rand();
  b[1] = .75*uni_rand();
}

int lowpass = FREQ_SIZE;
int band_lp = NB_BANDS;

void write_data_to_file(float * x, FILE * f)
{
    short tmp[FRAME_SIZE];
    for (int i=0;i<FRAME_SIZE;i++) tmp[i] = x[i];
    fwrite(tmp, sizeof(short),FRAME_SIZE,f);

}

int main(int argc, char **argv) {
  int i;
  int count=0;
  int clean_fileidx=0; //raoxy modify
  int noise_fileidx=0;
  int save_fileidx=0;
  static const float a_hp[2] = {-1.99599, 0.99600};
  static const float b_hp[2] = {-2, 1};
  float a_noise[2] = {0};
  float b_noise[2] = {0};
  float a_sig[2] = {0};
  float b_sig[2] = {0};
  float mem_hp_x[2]={0};
  float mem_hp_n[2]={0};
  float mem_resp_x[2]={0};
  float mem_resp_n[2]={0};
  float x[FRAME_SIZE];
  float n[FRAME_SIZE];
  float xn[FRAME_SIZE];
  float xn_processed[FRAME_SIZE];
  int vad_cnt=0;
  int gain_change_count=0;
  float speech_gain = 1, noise_gain = 1;
  char speech_fn[512], noise_fn[512], clean_raw_fn[512], noise_raw_fn[512],noisy_raw_fn[512],processed_raw_fn[512];
  FILE *f1, *f2, *f3, * f_clean, * f_noise, *f_noisy, *f_processed;
  int maxCount;
  DenoiseState *st;
  DenoiseState *de_noise_state;
  DenoiseState *noisy;
  st = rnnoise_create(NULL);
  de_noise_state = rnnoise_create(NULL);
  noisy = rnnoise_create(NULL);
  if (argc!=5) {
    fprintf(stderr, "usage: %s <speech directory> <noise directory> <count> <output>\n", argv[0]);
    return 1;
  }
  //f1 = fopen(argv[1], "r");
  //f2 = fopen(argv[2], "r");
  f3 = fopen(argv[4], "w");
  sprintf(speech_fn,"%s/clean_fileid_%d.wav",argv[1],clean_fileidx);
  sprintf(noise_fn,"%s/noise_fileid_%d.wav",argv[2],noise_fileidx);
  f1 = fopen(speech_fn, "r");
  f2 = fopen(noise_fn, "r");
  clean_fileidx = (clean_fileidx+1)%SPEECH_FILE_TOTAL;
  noise_fileidx = (noise_fileidx+1)%SPEECH_FILE_TOTAL;

  sprintf(clean_raw_fn,"clean_fileid_%d.raw", save_fileidx);
  sprintf(noise_raw_fn,"noise_fileid_%d.raw", save_fileidx);
  sprintf(noisy_raw_fn,"noisy_fileid_%d.raw", save_fileidx);
  sprintf(processed_raw_fn,"processed_fileid_%d.raw", save_fileidx);
  f_clean = fopen(clean_raw_fn, "w");
  f_noise = fopen(noise_raw_fn, "w");
  f_noisy = fopen(noisy_raw_fn, "w");
  f_processed = fopen(processed_raw_fn, "w");
  
  maxCount = atoi(argv[3]);
  
  short tmp[FRAME_SIZE];
  fread(tmp, sizeof(short), FRAME_SIZE, f1); // ignore the header and some data
  fread(tmp, sizeof(short), FRAME_SIZE, f2); // ignore the header and some data
  //what is the purpose for the following codes?
  //for(i=0;i<150;i++) {
  //  short tmp[FRAME_SIZE];
  //  fread(tmp, sizeof(short), FRAME_SIZE, f2);
  //}
  while (1) {
    kiss_fft_cpx X[FREQ_SIZE], Y[FREQ_SIZE], N[FREQ_SIZE], P[WINDOW_SIZE];
    kiss_fft_cpx Phat[WINDOW_SIZE];/*only for build*/
    float Ex[NB_BANDS], Ey[NB_BANDS], En[NB_BANDS], Ep[NB_BANDS];
    float Ephat[NB_BANDS], Ephaty[NB_BANDS]; /*only for build*/
    float Exp[NB_BANDS], Eyp[NB_BANDS], Ephatp[NB_BANDS];
    float Ln[NB_BANDS];
    float features[NB_FEATURES];
    float g[NB_BANDS];
    float r[NB_BANDS];
    short tmp[FRAME_SIZE];
    float vad=0;
    float E=0;
    if (count==maxCount) break;
    if ((count%1000)==0) fprintf(stderr, "%d\r", count);
    if (++gain_change_count > 2821) {
      //speech_gain = pow(10., (-40+(rand()%60))/20.);
      speech_gain = 1.0;
      //noise_gain = pow(10., (-30+(rand()%50))/20.);
      noise_gain = 1.0;
      if (rand()%10==0) noise_gain = 0;
      noise_gain *= speech_gain;
      if (rand()%10==0) speech_gain = 0;
      gain_change_count = 0;
      rand_resp(a_noise, b_noise);
      rand_resp(a_sig, b_sig);
      lowpass = FREQ_SIZE * 3000./24000. * pow(50., rand()/(double)RAND_MAX);
      //for (i=0;i<NB_BANDS;i++) {
      //  if (eband5ms[i]<<FRAME_SIZE_SHIFT > lowpass) {
      //    band_lp = i;
      //    break;
      //  }
      //}
    }
  
    if (speech_gain != 0) {
      fread(tmp, sizeof(short), FRAME_SIZE, f1);
      if (feof(f1)) {
         //rewind(f1);
          fclose(f1);
          sprintf(speech_fn,"%s/clean_fileid_%d.wav",argv[1],clean_fileidx);
          f1 = fopen(speech_fn, "r");
          clean_fileidx = (clean_fileidx+1)%SPEECH_FILE_TOTAL;
          fread(tmp, sizeof(short), FRAME_SIZE, f1); // ignore the header and some data
          fread(tmp, sizeof(short), FRAME_SIZE, f1);
      }
      for (i=0;i<FRAME_SIZE;i++) x[i] = speech_gain*tmp[i];
      for (i=0;i<FRAME_SIZE;i++) E += tmp[i]*(float)tmp[i];
    } else {
      for (i=0;i<FRAME_SIZE;i++) x[i] = 0;
      E = 0;
    }
    if (noise_gain!=0) {
      fread(tmp, sizeof(short), FRAME_SIZE, f2);
      if (feof(f2)) {
        //rewind(f2);
          fclose(f2);
          sprintf(noise_fn,"%s/noise_fileid_%d.wav",argv[2],noise_fileidx);
          f2 = fopen(noise_fn, "r");
          noise_fileidx = (noise_fileidx+1)%SPEECH_FILE_TOTAL;
          fread(tmp, sizeof(short), FRAME_SIZE, f2);// ignore the header and some data
          fread(tmp, sizeof(short), FRAME_SIZE, f2);
      }
      for (i=0;i<FRAME_SIZE;i++) n[i] = noise_gain*tmp[i];
    } else {
      for (i=0;i<FRAME_SIZE;i++) n[i] = 0;
    }
    write_data_to_file(x,f_clean);
    write_data_to_file(n,f_noise);

    biquad(x, mem_hp_x, x, b_hp, a_hp, FRAME_SIZE);
    biquad(x, mem_resp_x, x, b_sig, a_sig, FRAME_SIZE);
    biquad(n, mem_hp_n, n, b_hp, a_hp, FRAME_SIZE);
    biquad(n, mem_resp_n, n, b_noise, a_noise, FRAME_SIZE);
    for (i=0;i<FRAME_SIZE;i++) xn[i] = x[i] + n[i];

    write_data_to_file(xn,f_noisy);


    /*
    if (E > 1e9f) {
      vad_cnt=0;
    } else if (E > 1e8f) {Ephatyeriod);
    for (i=0;i<NB_BANDS;i++) {
      g[i] = sqrt((Ey[i]+1e-3)/(Ex[i]+1e-3));
      if (g[i] > 1) g[i] = 1;
      if (silence || i > band_lp) g[i] = -1;
      if (Ey[i] < 5e-2 && Ex[i] < 5e-2) g[i] = -1;
      if (vad==0 && noise_gain==0) g[i] = -1;
    }
    */
    
    int silence = compute_frame_features(noisy, Y, Phat/*not use*/, Ey, Ephat/*not use*/, Ephaty, features, xn);
    compute_frame_features(st, X, P, Ex, Ep, Exp, features, x);
    //calc_ideal_gain(Ex, Ey, g);
    for (i=0;i<NB_BANDS;i++) {
      g[i] = sqrt((Ex[i]+1e-3)/(Ey[i]+1e-3));
      if (g[i] > 1) {
        g[i] = 1;
        printf("g[%d] >1, g[i]:%f\n", i, g[i]);
      }
      if (silence || i > band_lp) g[i] = 0;
      if (Ey[i] < 5e-2 && Ex[i] < 5e-2) g[i] = 0;
      if(g[i]<0){ printf("g[%d] < 0 detected\n", i);}
      if(isnan(g[i])) {printf("g[%d] is Nan detected Ex[i]:%f, Ey[i]:%f\n", i, Ex[i], Ey[i]);}
      //if (vad==0 && noise_gain==0) g[i] = -1;
    }
    
    //compute_band_corr(Eyp, Y, P);
    //for (i=0;i<NB_BANDS;i++) Eyp[i] = Eyp[i]/sqrt(.001+Ey[i]*Ep[i]);
    estimate_phat_corr(common, Ephaty, Ephatp);
    filter_strength_calc(Exp, Ephaty, Ephatp, r);
    adjust_gain_strength_by_condition(common, Ephatp, Ephaty, Exp, g, r);

#if 0
    fwrite(features, sizeof(float), NB_FEATURES, stdout);
    fwrite(g, sizeof(float), NB_BANDS, stdout);
    fwrite(Ln, sizeof(float), NB_BANDS, stdout);
    fwrite(&vad, sizeof(float), 1, stdout);
#endif
    //fwrite(features, sizeof(float), NB_FEATURES, stdout);
    fwrite(Ey, sizeof(float), NB_BANDS, f3);//Y(l+M)
    fwrite(Ephaty, sizeof(float), NB_BANDS, f3);//pitch coherence
    
    float T = noisy->last_period/(PITCH_MAX_PERIOD-3*PITCH_MIN_PERIOD);
    float pitchcorr = noisy->pitch_corr;
    fwrite(&T, sizeof(float), 1, f3);//pitch
    fwrite(&pitchcorr, sizeof(float), 1, f3);//pitch correlation
    
    fwrite(r, sizeof(float), NB_BANDS, f3);//filtering strength
    fwrite(g, sizeof(float), NB_BANDS, f3);//gain

    //process the noisy file using the compuited g and r
    rnnoise_process_frame_for_validate_data(de_noise_state, xn_processed, xn, g, r);
    write_data_to_file(xn_processed, f_processed);
    //fwrite(xn_processed,sizeof(float), FRAME_SIZE,f_processed);
        
    count++;
    //save the files every 30s
    if(count % 3000 == 0){
      //close the files
      fclose(f_clean);
      fclose(f_noise);
      fclose(f_noisy);
      fclose(f_processed);

      //open new files
      save_fileidx++;
      sprintf(clean_raw_fn,"clean_fileid_%d.raw", save_fileidx);
      sprintf(noise_raw_fn,"noise_fileid_%d.raw", save_fileidx);
      sprintf(noisy_raw_fn,"noisy_fileid_%d.raw", save_fileidx);
      sprintf(processed_raw_fn,"processed_fileid_%d.raw", save_fileidx);
      f_clean = fopen(clean_raw_fn, "w");
      f_noise = fopen(noise_raw_fn, "w");
      f_noisy = fopen(noisy_raw_fn, "w");
      f_processed = fopen(processed_raw_fn, "w");
    }

    for (i=0;i<NB_BANDS;i++) {
     if(isnan(Ey[i])) {printf("Ey[%d] is Nan detected \n", i);}
     if(isnan(Ephaty[i])) {printf("Ephaty[%d] is Nan detected\n", i);}
     if(isnan(g[i])) {printf("g[%d] is Nan detected \n", i);}
     if(isnan(r[i])) {printf("r[%d] is Nan detected\n", i);}
     if(g[i] < 0 || g[i] > 1) printf("g[%d] is invalid:%f\n", i,g[i]);
     if(r[i] < 0 || r[i] > 1) printf("r[%d] is invalid:%f\n", i,r[i]);
   }
   if(isnan(T)) {printf("T is Nan detected \n");}
   if(isnan(pitchcorr)) {printf("pitchcorr is Nan detected \n");} 

  }
  fprintf(stderr, "matrix size: %d x %d\n", count, 2 + 4*NB_BANDS);
  fclose(f1);
  fclose(f2);
  fclose(f3);
  fclose(f_clean);
  fclose(f_noise);
  fclose(f_noisy);
  fclose(f_processed);
  return 0;
}

#endif
