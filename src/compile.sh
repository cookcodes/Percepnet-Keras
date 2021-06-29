#!/bin/sh

gcc -DTRAINING=1 -Wall -W -O3 -g -I../include denoise.c kiss_fft.c pitch.c celt_lpc.c rnn.c rnn_data.c -o denoise_training -lm
gcc -DTEST_GR=1 -Wall -W -O3 -g -I../include denoise.c kiss_fft.c pitch.c celt_lpc.c rnn.c rnn_data.c wav.c -o test_gr -lm