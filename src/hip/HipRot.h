#ifndef HIPROT_H
#define HIPROT_H

#include "../default.h"
#include <hip/hip_runtime.h>
#include "HipVariables.h"
#include "HipKernel.h"
#include "HipTextures.h"
#include "HipContext.h"


using namespace Hip;

class HipRot
{
private:
	HipKernel* rotVol;
	HipKernel* shift;
	HipKernel* rotVolCplx;

	Hip::HipContext* ctx;
	int volSize;
	dim3 blockSize;
	dim3 gridSize;

	float oldphi, oldpsi, oldtheta;

	HipArray3D shiftTex;
	HipArray3D dataTex;
	HipArray3D dataTexCplx;

	hipStream_t stream;

	void runShiftKernel(HipDeviceVariable& d_odata, float3 shiftVal);
	void runRotKernel(HipDeviceVariable& d_odata, float rotMat[3][3]);
	void runRotCplxKernel(HipDeviceVariable& d_odata, float rotMat[3][3]);

	void computeRotMat(float phi, float psi, float theta, float rotMat[3][3]);
	void multiplyRotMatrix(float m1[3][3], float m2[3][3], float out[3][3]);
public:

	HipRot(int aVolSize, hipStream_t aStream, Hip::HipContext* context, bool linearInterpolation);

	void SetTextureShift(HipDeviceVariable& d_idata);
	void SetTexture(HipDeviceVariable& d_idata);
	void SetTextureCplx(HipDeviceVariable& d_idata);

	void Shift(HipDeviceVariable& d_odata, float3 shiftVal);
	void Rot(HipDeviceVariable& d_odata, float phi, float psi, float theta);
	void RotCplx(HipDeviceVariable& d_odata, float phi, float psi, float theta);

	void SetOldAngles(float aPhi, float aPsi, float aTheta);
};

#endif
