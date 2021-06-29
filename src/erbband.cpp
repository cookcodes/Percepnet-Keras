#include "erbband.h"
#include <iostream>
using namespace std;

#define FRAME_SIZE_SHIFT 2
#define FRAME_SIZE (120<<FRAME_SIZE_SHIFT)
#define WINDOW_SIZE (2*FRAME_SIZE)

#define NB_BANDS 34

ERBBand *erb_band = new ERBBand(WINDOW_SIZE, NB_BANDS, 0/*low_freq*/, 20000/*high_freq*/);

int main()
{
  for (int i=0;i<NB_BANDS;i++)
  {
    int j;
    int band_size;
    //band_size = (eband5ms[i+1]-eband5ms[i])<<FRAME_SIZE_SHIFT;
    int low_nfft_idx = erb_band->filters[i].first.first;
    int high_nfft_idx = erb_band->filters[i].first.second;
    cout<<"float erb_weights_"<<i<<"[]={";
    for(j=low_nfft_idx; j<high_nfft_idx; j++){
      cout<<erb_band->filters[i].second[j-low_nfft_idx]<<",";
    }
    cout<<"}\n";
  }

  for (int i=0;i<NB_BANDS;i++)
  {
    int j;
    int band_size;
    //band_size = (eband5ms[i+1]-eband5ms[i])<<FRAME_SIZE_SHIFT;
    int low_nfft_idx = erb_band->filters[i].first.first;
    int high_nfft_idx = erb_band->filters[i].first.second;
    cout<<"    {\n        "<<low_nfft_idx<<",\n        "<<high_nfft_idx<<",\n        erb_weights_"<<i<<",\n    },";
  }

  return 0;
}
