#ifndef CTFMODEL_H
#define CTFMODEL_H

#include "default.h"
class CTFModel
{
 private:
  float _defocus;
  float _openingAngle;
  float _ampContrast;
  float _phaseContrast;
  float _pixelsize;
  float _pixelcount;
  float _maxFreq;
  float _freqStepSize;
  float* _CTFImage;
  float* Xvalues;
  float* Yvalues;
  float* Xaxis;
  float* _envelope;
  float applyScatteringProfile;
  float applyEnvelopeFunction;

  const static float _voltage = 300;
  const static float h = 6.63E-34; //Planck's quantum
  const static float c = 3.00E+08; //Light speed
  const static float Cs = 2.7 * 0.001;
  const static float Cc = 2.7 * 0.001;

  const static float PhaseShift = 0;
  const static float EnergySpread = 0.7; //eV
  const static float E0 = 511; //keV
  float RelativisticCorrectionFactor;// = (1 + _voltage / (E0 * 1000))/(1 + ((_voltage*1000) / (2 * E0 * 1000)));
  float H;// = (Cc * EnergySpread * RelativisticCorrectionFactor) / (_voltage * 1000);

  const static float a1 = 1.494; //Scat.Profile Carbon Amplitude 1
  const static float a2 = 0.937; //Scat.Profile Carbon Amplitude 2
  const static float b1 = 23.22 * 1E-20; //Scat.Profile Carbon Halfwidth 1
  const static float b2 = 3.79 * 1E-20;  //Scat.Profile Carbon Halfwidth 2
  float lambda;
  bool _absolut;

 public:
  CTFModel(float defocus, float pixelsize, float pixelcount, float openingAngle, float ampContrast);

  float* GetCTF();

  float* GetCTFImage();

  void SetDefocus(float value);

};

#endif // CTFMODEL_H
