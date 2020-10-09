#ifndef PROJECTION_H
#define PROJECTION_H

#include "default.h"
#include "io/MRCFile.h"
#include "io/Dm4FileStack.h"
#include "io/MarkerFile.h"
#include "utils/Matrix.h"
#include "hip_kernels/Constants.h"
#include "Volume.h"

enum ProjectionListType
{
	PLT_NORMAL,
	PLT_RANDOM,
	PLT_RANDOM_START_ZERO_TILT,
	PLT_RANDOM_MIDDLE_PROJ_TWICE
};

class Projection
{
protected:
	ProjectionSource* mrc;
	MarkerFile* markers;
	float2* extraShifts;

public:
	Projection(ProjectionSource* aMrc, MarkerFile* aMarkers);
	~Projection();

	dim3 GetDimension();
	int GetWidth();
	int GetHeight();
	int GetMaxDimension();
	float3 GetPosition(uint aIndex);
	float3 GetPixelUPitch(uint aIndex);  //c_zPitch
	float3 GetPixelVPitch(uint aIndex);  //c_yPitch
	float3 GetNormalVector(uint aIndex);
	void GetDetectorMatrix(uint aIndex, float aMatrix[16], float os);
	int GetMinimumTiltIndex();
	float2 GetMinimumTiltShift();
	float2 GetMeanShift();
	float2 GetMedianShift();
	float GetMean(float* data);
	float GetMean(int index);
	void Normalize(float* data, float mean);
	Matrix<float> RotateMatrix(uint aIndex, Matrix<float>& matrix);
	void CreateProjectionIndexList(ProjectionListType type, int* projectionCount, int** indexList);
	float GetPixelSize(int index);

	void ComputeHitPoints(Volume<unsigned short>& vol, uint index, int2& pA, int2& pB, int2& pC, int2& pD);
	void ComputeHitPoints(Volume<float>& vol, uint index, int2& pA, int2& pB, int2& pC, int2& pD);
	void ComputeHitPoint(float posX, float posY, float posZ, uint index, int2& pA);

	float2 GetExtraShift(size_t index);
	void SetExtraShift(size_t index, float2 extraShift);
	void AddExtraShift(size_t index, float2 extraShift);
};

#endif
