#include "HipRot.h"

const hipChannelFormatDesc float_desc = myhipCreateChannelDesc( 32, 0, 0, 0, hipChannelFormatKindFloat);
const hipChannelFormatDesc float2_desc = myhipCreateChannelDesc( 32, 32, 0, 0, hipChannelFormatKindFloat);

HipRot::HipRot(int aVolSize, hipStream_t aStream, Hip::HipContext* context, bool linearInterpolation)
	: volSize(aVolSize), stream(aStream), ctx(context), blockSize(32, 16, 1),
	  gridSize(aVolSize / 32, aVolSize / 16, aVolSize),
	  shiftTex(float_desc, aVolSize, aVolSize, aVolSize, 0),
	  dataTex(float_desc, aVolSize, aVolSize, aVolSize, 0),
	  dataTexCplx(float2_desc, aVolSize, aVolSize, aVolSize, 0),
  	  /*shiftTex(CU_AD_FORMAT_FLOAT, aVolSize, aVolSize, aVolSize, 1, 0),
	  dataTex(CU_AD_FORMAT_FLOAT, aVolSize, aVolSize, aVolSize, 1, 0),
	  dataTexCplx(CU_AD_FORMAT_FLOAT, aVolSize, aVolSize, aVolSize, 2, 0),*/
	  oldphi(0), oldpsi(0), oldtheta(0)
{
	hipModule_t kernelModulue = ctx->LoadModule("basicKernels.ptx");

	shift = new HipKernel("shift", kernelModulue);
	rotVol = new HipKernel("rot3d", kernelModulue);
	rotVolCplx = new HipKernel("rot3dCplx", kernelModulue);

	CUfilter_mode filter = linearInterpolation ? CU_TR_FILTER_MODE_LINEAR : CU_TR_FILTER_MODE_POINT;

	CudaTextureArray3D shiftTex2(rotVol, "texShift", CU_TR_ADDRESS_MODE_WRAP, CU_TR_ADDRESS_MODE_WRAP,
		CU_TR_ADDRESS_MODE_WRAP, CU_TR_FILTER_MODE_LINEAR, CU_TRSF_NORMALIZED_COORDINATES, &shiftTex);
	CudaTextureArray3D tex(rotVol, "texVol", CU_TR_ADDRESS_MODE_CLAMP, CU_TR_ADDRESS_MODE_CLAMP,
		CU_TR_ADDRESS_MODE_CLAMP, filter, 0, &dataTex);
	CudaTextureArray3D texCplx(rotVolCplx, "texVolCplx", CU_TR_ADDRESS_MODE_CLAMP, CU_TR_ADDRESS_MODE_CLAMP,
		CU_TR_ADDRESS_MODE_CLAMP, filter, 0, &dataTexCplx);
}


void HipRot::SetTexture(HipDeviceVariable& d_idata)
{
	dataTex.CopyFromDeviceToArray(d_idata);
}


void HipRot::SetTextureShift(HipDeviceVariable& d_idata)
{
	shiftTex.CopyFromDeviceToArray(d_idata);
}


void HipRot::SetTextureCplx(HipDeviceVariable& d_idata)
{
	dataTexCplx.CopyFromDeviceToArray(d_idata);
}


void HipRot::Rot(HipDeviceVariable& d_odata, float phi, float psi, float theta)
{
	float rotMat1[3][3];
	float rotMat2[3][3];
	float rotMat[3][3];
	computeRotMat(oldphi, oldpsi, oldtheta, rotMat1);
	computeRotMat(phi, psi, theta, rotMat2);
	multiplyRotMatrix(rotMat2, rotMat1, rotMat);

	runRotKernel(d_odata, rotMat);
}


void HipRot::Shift(HipDeviceVariable& d_odata, float3 shiftVal)
{
	runShiftKernel(d_odata, shiftVal);
}


void HipRot::RotCplx(HipDeviceVariable& d_odata, float phi, float psi, float theta)
{
	float rotMat1[3][3];
	float rotMat2[3][3];
	float rotMat[3][3];
	computeRotMat(oldphi, oldpsi, oldtheta, rotMat1);
	computeRotMat(phi, psi, theta, rotMat2);
	multiplyRotMatrix(rotMat2, rotMat1, rotMat);

	runRotCplxKernel(d_odata, rotMat);
}


void HipRot::computeRotMat(float phi, float psi, float theta, float rotMat[3][3])
{
	int i, j;
	float sinphi, sinpsi, sintheta;	/* sin of rotation angles */
	float cosphi, cospsi, costheta;	/* cos of rotation angles */


	float angles[] = { 0, 30, 45, 60, 90, 120, 135, 150, 180, 210, 225, 240, 270, 300, 315, 330 };
	float angle_cos[16];
	float angle_sin[16];

	angle_cos[0]=1.0f;
	angle_cos[1]=sqrt(3.0f)/2.0f;
	angle_cos[2]=sqrt(2.0f)/2.0f;
	angle_cos[3]=0.5f;
	angle_cos[4]=0.0f;
	angle_cos[5]=-0.5f;
	angle_cos[6]=-sqrt(2.0f)/2.0f;
	angle_cos[7]=-sqrt(3.0f)/2.0f;
	angle_cos[8]=-1.0f;
	angle_cos[9]=-sqrt(3.0f)/2.0f;
	angle_cos[10]=-sqrt(2.0f)/2.0f;
	angle_cos[11]=-0.5f;
	angle_cos[12]=0.0f;
	angle_cos[13]=0.5f;
	angle_cos[14]=sqrt(2.0f)/2.0f;
	angle_cos[15]=sqrt(3.0f)/2.0f;
	angle_sin[0]=0.0f;
	angle_sin[1]=0.5f;
	angle_sin[2]=sqrt(2.0f)/2.0f;
	angle_sin[3]=sqrt(3.0f)/2.0f;
	angle_sin[4]=1.0f;
	angle_sin[5]=sqrt(3.0f)/2.0f;
	angle_sin[6]=sqrt(2.0f)/2.0f;
	angle_sin[7]=0.5f;
	angle_sin[8]=0.0f;
	angle_sin[9]=-0.5f;
	angle_sin[10]=-sqrt(2.0f)/2.0f;
	angle_sin[11]=-sqrt(3.0f)/2.0f;
	angle_sin[12]=-1.0f;
	angle_sin[13]=-sqrt(3.0f)/2.0f;
	angle_sin[14]=-sqrt(2.0f)/2.0f;
	angle_sin[15]=-0.5f;

	for (i=0, j=0 ; i<16; i++)
	{
		if (angles[i] == phi )
		{
			cosphi = angle_cos[i];
			sinphi = angle_sin[i];
			j = 1;
		}
	}

	if (j < 1)
	{
		phi = phi * (float)M_PI / 180.0f;
		cosphi=cos(phi);
		sinphi=sin(phi);
	}

	for (i=0, j=0 ; i<16; i++)
	{
		if (angles[i] == psi )
		{
			cospsi = angle_cos[i];
			sinpsi = angle_sin[i];
			j = 1;
		}
	}

	if (j < 1)
	{
		psi = psi * (float)M_PI / 180.0f;
		cospsi=cos(psi);
		sinpsi=sin(psi);
	}

	for (i=0, j=0 ; i<16; i++)
	{
		if (angles[i] == theta )
		{
		   costheta = angle_cos[i];
		   sintheta = angle_sin[i];
		   j = 1;
		}
	}

	if (j < 1)
	{
		theta = theta * (float)M_PI / 180.0f;
		costheta=cos(theta);
		sintheta=sin(theta);
	}

	/* calculation of rotation matrix */

	rotMat[0][0] = cospsi*cosphi-costheta*sinpsi*sinphi;
	rotMat[1][0] = sinpsi*cosphi+costheta*cospsi*sinphi;
	rotMat[2][0] = sintheta*sinphi;
	rotMat[0][1] = -cospsi*sinphi-costheta*sinpsi*cosphi;
	rotMat[1][1] = -sinpsi*sinphi+costheta*cospsi*cosphi;
	rotMat[2][1] = sintheta*cosphi;
	rotMat[0][2] = sintheta*sinpsi;
	rotMat[1][2] = -sintheta*cospsi;
	rotMat[2][2] = costheta;
}


void HipRot::multiplyRotMatrix(float m1[3][3], float m2[3][3], float out[3][3])
{
	out[0][0] = m1[0][0] * m2[0][0] + m1[1][0] * m2[0][1] + m1[2][0] * m2[0][2];
    out[1][0] = m1[0][0] * m2[1][0] + m1[1][0] * m2[1][1] + m1[2][0] * m2[1][2];
    out[2][0] = m1[0][0] * m2[2][0] + m1[1][0] * m2[2][1] + m1[2][0] * m2[2][2];
    out[0][1] = m1[0][1] * m2[0][0] + m1[1][1] * m2[0][1] + m1[2][1] * m2[0][2];
    out[1][1] = m1[0][1] * m2[1][0] + m1[1][1] * m2[1][1] + m1[2][1] * m2[1][2];
    out[2][1] = m1[0][1] * m2[2][0] + m1[1][1] * m2[2][1] + m1[2][1] * m2[2][2];
    out[0][2] = m1[0][2] * m2[0][0] + m1[1][2] * m2[0][1] + m1[2][2] * m2[0][2];
    out[1][2] = m1[0][2] * m2[1][0] + m1[1][2] * m2[1][1] + m1[2][2] * m2[1][2];
    out[2][2] = m1[0][2] * m2[2][0] + m1[1][2] * m2[2][1] + m1[2][2] * m2[2][2];
}


void HipRot::runRotKernel(HipDeviceVariable& d_odata, float rotMat[3][3])
{
    hipDeviceptr_t out_dptr = d_odata.GetDevicePtr();

	float3 rotMat0 = make_float3(rotMat[0][0], rotMat[0][1], rotMat[0][2]);
	float3 rotMat1 = make_float3(rotMat[1][0], rotMat[1][1], rotMat[1][2]);
	float3 rotMat2 = make_float3(rotMat[2][0], rotMat[2][1], rotMat[2][2]);

    void** arglist = (void**)new void*[5];

    arglist[0] = &volSize;
    arglist[1] = &rotMat0;
    arglist[2] = &rotMat1;
    arglist[3] = &rotMat2;
    arglist[4] = &out_dptr;

    hipSafeCall(hipModuleLaunchKernel(rotVol->GetHipFunction(), gridSize.x, gridSize.y,
		gridSize.z, blockSize.x, blockSize.y, blockSize.z, 0, stream, arglist,NULL));

    delete[] arglist;
}


void HipRot::runShiftKernel(HipDeviceVariable& d_odata, float3 shiftVal)
{
    hipDeviceptr_t out_dptr = d_odata.GetDevicePtr();

    void** arglist = (void**)new void*[3];

    arglist[0] = &volSize;
    arglist[1] = &out_dptr;
    arglist[2] = &shiftVal;

    hipSafeCall(hipModuleLaunchKernel(shift->GetHipFunction(), gridSize.x, gridSize.y,
		gridSize.z, blockSize.x, blockSize.y, blockSize.z, 0, stream, arglist,NULL));

    delete[] arglist;
}


void HipRot::runRotCplxKernel(HipDeviceVariable& d_odata, float rotMat[3][3])
{
	hipDeviceptr_t out_dptr = d_odata.GetDevicePtr();

	float3 rotMat0 = make_float3(rotMat[0][0], rotMat[0][1], rotMat[0][2]);
	float3 rotMat1 = make_float3(rotMat[1][0], rotMat[1][1], rotMat[1][2]);
	float3 rotMat2 = make_float3(rotMat[2][0], rotMat[2][1], rotMat[2][2]);

    void** arglist = (void**)new void*[5];

    arglist[0] = &volSize;
    arglist[1] = &rotMat0;
    arglist[2] = &rotMat1;
    arglist[3] = &rotMat2;
    arglist[4] = &out_dptr;

    hipSafeCall(hipModuleLaunchKernel(rotVolCplx->GetHipFunction(), gridSize.x, gridSize.y,
		gridSize.z, blockSize.x, blockSize.y, blockSize.z, 0, stream, arglist,NULL));

    delete[] arglist;
}


void HipRot::SetOldAngles(float aPhi, float aPsi, float aTheta)
{
	oldphi = aPhi;
	oldpsi = aPsi;
	oldtheta = aTheta;
}

