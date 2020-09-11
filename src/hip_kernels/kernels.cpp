#define EPS (0.000001f)

extern "C" __global__ void MulReal(int size, float in, float* outVol){
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;	
	const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;	

	outVol[z * size * size + y * size + x] = outVol[z * size * size + y * size + x] * in;
}

extern "C" __global__ void Mul(int size, float in, float2* outVol){
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;	
	const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;	
	
	float2 temp = outVol[z * size * size + y * size + x];
	temp.x *= in;
	temp.y *= in;
	outVol[z * size * size + y * size + x] = temp;
}

extern "C" __global__ void Correl(int size, float2* inVol, float2* outVol){
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;	
	const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;	
	
	float2 o = outVol[z * size * size + y * size + x];
	float2 i = inVol[z * size * size + y * size + x];
	float2 erg;
	erg.x = (o.x * i.x) + (o.y * i.y);
	erg.y = (o.x * i.y) - (o.y * i.x);
	outVol[z * size * size + y * size + x] = erg;
}

extern "C" __global__ void Correl_R2C(int size, float2* inVol, float2* outVol){
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;	
	const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;	
	
	float2 o = outVol[z * size * (size/2+1) + y * (size/2+1) + x];
	float2 i = inVol[z * size * (size/2+1) + y * (size/2+1) + x];
	float2 erg;
	erg.x = (o.x * i.x) + (o.y * i.y);
	erg.y = (o.x * i.y) - (o.y * i.x);
	outVol[z * size * (size/2+1) + y * (size/2+1) + x] = erg;
}

extern "C" __global__ void Conv(int size, float2* inVol, float2* outVol){
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;	
	const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;	
	
	float2 o = outVol[z * size * size + y * size + x];
	float2 i = inVol[z * size * size + y * size + x];
	float2 erg;
	erg.x = (o.x * i.x) - (o.y * i.y);
	erg.y = (o.x * i.y) + (o.y * i.x);
	outVol[z * size * size + y * size + x] = erg;
}

extern "C" __global__ void Conv_R2C(int size, float2* inVol, float2* outVol){
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;	
	const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;	
	
	float2 o = outVol[z * size * (size/2+1) + y * (size/2+1) + x];
	float2 i = inVol[z * size * (size/2+1) + y * (size/2+1) + x];
	float2 erg;
	erg.x = (o.x * i.x) - (o.y * i.y);
	erg.y = (o.x * i.y) + (o.y * i.x);
	outVol[z * size * (size/2+1) + y * (size/2+1) + x] = erg;
}

extern "C" __global__ void Energynorm(int size, float2* particle, float2* partSqr, float2* cccMap, float* energyRef, float* nVox){
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;	
	const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;	
	
	float part = particle[z * size * size + y * size + x].x; 
	float energyLocal = partSqr[z * size * size + y * size + x].x; 
	
	float2 erg;
	erg.x = 0;
	erg.y = 0;

	energyLocal -= part * part / nVox[0];
	energyLocal = sqrt(energyLocal) * sqrt(energyRef[0]);

	if (energyLocal > EPS)
	{
		erg.x = cccMap[z * size * size + y * size + x].x / energyLocal;
	}

	cccMap[z * size * size + y * size + x] = erg;
}

extern "C" __global__ void Energynorm_R2C(int size, float* particle, float* partSqr, float* cccMap, float* energyRef, float* nVox, float2* temp, float* ccMask){
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;	
	const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;	
	
	float part = particle[z * size * size + y * size + x]; 
	float energyLocal = partSqr[z * size * size + y * size + x]; 
	
	float erg = 0;
	
	energyLocal -= part * part / nVox[0];
	energyLocal = sqrt(energyLocal) * sqrt(energyRef[0]);

	if (energyLocal > EPS)
	{
		erg = cccMap[z * size * size + y * size + x] / energyLocal;
	}

	int i = (x + size / 2) % size;
	int j = (y + size / 2) % size;
	int k = (z + size / 2) % size;

	cccMap[z * size * size + y * size + x] = erg;
	erg *= ccMask[k * size * size + j * size + i];
	temp[k * size * size + j * size + i].x = erg;
}

extern "C" __global__ void Sub(int size, float* inVol, float* outVol, float val){
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;	
	const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;	
	
	outVol[z * size * size + y * size + x] = inVol[z * size * size + y * size + x] - val;
}

extern "C" __global__ void Sub_R2C(int size, float* inVol, float* outVol, float* h_sum, float val){
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;	
	const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;	
	
	outVol[z * size * size + y * size + x] = inVol[z * size * size + y * size + x] - (h_sum[0]/val);
}

extern "C" __global__ void SqrSub_R2C(int size, float* inVol, float* outVol, float* h_sum, float val){
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;	
	const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;	
	
	outVol[z * size * size + y * size + x] = inVol[z * size * size + y * size + x]*inVol[z * size * size + y * size + x] - (h_sum[0]/val);
}

extern "C" __global__ void SubAndSqrSub(int size, float* inVol, float* outVol, float* h_sum, float val){
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;	
	const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;	
	
	float tmp = inVol[z * size * size + y * size + x] - (h_sum[0]/val);
	outVol[z * size * size + y * size + x] = tmp * tmp;
	inVol[z * size * size + y * size + x] = tmp;
}

extern "C" __global__ void Binarize(int size, float* inVol, float* outVol){
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;	
	const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;	
	
	outVol[z * size * size + y * size + x] = inVol[z * size * size + y * size + x] > 0.5f ? 1.0f : 0.0f;
}

extern "C" __global__ void SubCplx(int size, float2* inVol, float2* outVol, float* subval, float* divVal){
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;	
	const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;	
	
	float2 temp = inVol[z * size * size + y * size + x];
	temp.x -= subval[0] / divVal[0];
	outVol[z * size * size + y * size + x] = temp;
}

extern "C" __global__ void SubCplx_R2C(int size, float* inVol, float* outVol, float* subval, float* divVal){
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;	
	const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;	

	outVol[z * size * (size) + y * (size) + x] = inVol[z * size * size + y * (size) + x] - subval[0] / divVal[0];
}

extern "C" __global__ void FFTShift(int size, float2* volIn, float2* volOut){
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;	
	const int z = blockIdx.z * blockDim.z + threadIdx.z;	
	
	int i = (x + size / 2) % size;
	int j = (y + size / 2) % size;
	int k = (z + size / 2) % size;
	// int i, j, k;
	// x <= size/2 ? i = x+size/2 : i= x-size/2;
	// y <= size/2 ? j = y+size/2 : j= y-size/2;
	// z <= size/2 ? k = z+size/2 : k= z-size/2;

	float2 temp = volIn[k * size * size + j * size + i]; 
	volOut[z * size * size + y * size + x] = temp;
}

extern "C" __global__ void FFTShiftReal(int size, float* volIn, float* volOut){
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;	
	const int z = blockIdx.z * blockDim.z + threadIdx.z;	
	
	int i = (x + size / 2) % size;
	int j = (y + size / 2) % size;
	int k = (z + size / 2) % size;


	float temp = volIn[k * size * size + j * size + i]; 
	volOut[z * size * size + y * size + x] = temp;
}

extern "C"
__global__ void MulVol(int size, float* inVol, float2* outVol)
{
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;	
	const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;	
	
	float2 temp = outVol[z * size * size + y * size + x];
	temp.x *= inVol[z * size * size + y * size + x];
	temp.y *= inVol[z * size * size + y * size + x];
	outVol[z * size * size + y * size + x] = temp;
}

extern "C" __global__ void MulVol_R2C(int size, float* inVol, float2* outVol){
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;	
	const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;	
	
	float2 temp = outVol[z * size * (size / 2 + 1)  + y * (size / 2 + 1) + x];
	temp.x *= inVol[z * size * size + y * size + x];
	temp.y *= inVol[z * size * size + y * size + x];
	outVol[z * size * (size / 2 + 1) + y * (size / 2 + 1) + x] = temp;
}

extern "C" __global__ void MulVolReal(int size, float* inVol, float* outVol){
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;	
	const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;	
	
	float temp = outVol[z * size * size  + y * size + x];
	temp *= inVol[z * size * size + y * size + x];
	outVol[z * size * size  + y * size + x] = temp;
}

extern "C" __global__ void BandpassFFTShift(int size, float2* vol, float rDown, float rUp, float smooth){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;	
	int z = blockIdx.z * blockDim.z + threadIdx.z;	
	
	int i = (x + size / 2) % size;
	int j = (y + size / 2) % size;
	int k = (z + size / 2) % size;

	float2 temp = vol[z * size * size + y * size + x];

	//use squared smooth for Gaussian
	smooth = smooth * smooth;

	float center = size / 2;
	float3 vox = make_float3(i - center, j - center, k - center);

	float dist = sqrt(vox.x * vox.x + vox.y * vox.y + vox.z * vox.z);
	float scf = (dist - rUp) * (dist - rUp);
	smooth > 0 ? scf = exp(-scf/smooth) : scf = 0;

	if (dist > rUp)
	{
		temp.x *= scf;
		temp.y *= scf;
	}
	
	scf = (dist - rDown) * (dist - rDown);
	smooth > 0 ? scf = exp(-scf/smooth) : scf = 0;
	
	if (dist < rDown)
	{
		temp.x *= scf;
		temp.y *= scf;
	}
	

	vol[z * size * size + y * size + x] = temp;
}

extern "C" __global__ void BandpassFFTShift_R2C(int size, float2* vol, float rDown, float rUp, float smooth){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;	
	int z = blockIdx.z * blockDim.z + threadIdx.z;	
	
	int i = (x + size / 2) % size;// x; // (x + size / 2) % (size /2);
	int j = (y + size / 2) % size;
	int k = (z + size / 2) % size;

	float2 temp = vol[z * size * (size/2+1) + y * (size/2+1) + x];

	//use squared smooth for Gaussian
	smooth = smooth * smooth;

	float center = size / 2;
	float3 vox = make_float3(i - center, j - center, k - center);

	float dist = sqrt(vox.x * vox.x + vox.y * vox.y + vox.z * vox.z);
	float scf = (dist - rUp) * (dist - rUp);
	smooth > 0 ? scf = exp(-scf/smooth) : scf = 0;

	if (dist > rUp)
	{
		temp.x *= scf;
		temp.y *= scf;
	}
	
	scf = (dist - rDown) * (dist - rDown);
	smooth > 0 ? scf = exp(-scf/smooth) : scf = 0;
	
	if (dist < rDown)
	{
		temp.x *= scf;
		temp.y *= scf;
	}

	vol[z * size * (size/2+1) + y * (size/2+1) + x] = temp;
}


// extern "C" __global__ void ApplyWedgeFilterBandpass(int size, float* manipulation, float* wedge, float* filter, float2* vol, float rDown, float rUp, float smooth, bool useFilterVolume){
// 	int x = blockIdx.x * blockDim.x + threadIdx.x;
// 	int y = blockIdx.y * blockDim.y + threadIdx.y;	
// 	int z = blockIdx.z * blockDim.z + threadIdx.z;	
	
// 	float2 temp = vol[z * size * size + y * size + x]; // * wedge[z * size * size + y * size + x];
// 	float l_wedge = wedge[z * size * size + y * size + x];
// 	temp.x *= l_wedge;
// 	temp.y *= l_wedge;
// 	if (useFilterVolume){
// 		l_wedge = filter[z * size * size + y * size + x];
// 		temp.x *= filter;
// 		temp.y *= filter;
// 	}
// 	else
// 	{
// 		int i = (x + size / 2) % size;
// 		int j = (y + size / 2) % size;
// 		int k = (z + size / 2) % size;

// 		//use squared smooth for Gaussian
// 		smooth = smooth * smooth;

// 		float center = size / 2;
// 		float3 vox = make_float3(i - center, j - center, k - center);

// 		float dist = sqrt(vox.x * vox.x + vox.y * vox.y + vox.z * vox.z);
// 		float scf = (dist - rUp) * (dist - rUp);
// 		smooth > 0 ? scf = exp(-scf/smooth) : scf = 0;

// 		if (dist > rUp)
// 		{
// 			manipulation_factor *= scf;
// 		}
		
// 		scf = (dist - rDown) * (dist - rDown);
// 		smooth > 0 ? scf = exp(-scf/smooth) : scf = 0;
		
// 		if (dist < rDown)
// 		{
// 			manipulation_factor *= scf;
// 		}
// 	}
// 	manipulation[z * size * size + y * size + x] = manipulation_factor;
// 	temp.x *= manipulation_factor;
// 	temp.y *= manipulation_factor;
// 	vol[z * size * size + y * size + x] = temp;
// }

extern "C" __global__ void ApplyWedgeFilterBandpass_RC(int size, float* wedge, float* filter, float2* vol, float rDown, float rUp, float smooth, bool useFilterVolume){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;	
	int z = blockIdx.z * blockDim.z + threadIdx.z;	
	
	float2 temp = vol[z * size * (size / 2 + 1) + y * (size / 2 + 1) + x]; // * wedge[z * size * size + y * size + x];
	
	temp.x *= wedge[z * size * size + y * size + x];
	temp.y *= wedge[z * size * size + y * size + x];

	if (useFilterVolume){
		temp.x *= filter[z * size * size + y * size + x];
		temp.y *= filter[z * size * size + y * size + x];
	}
	else
	{
		int i = (size / 2) - x; // (x + size / 2) % (size /2);
		int j = (y + size / 2) % size;
		int k = (z + size / 2) % size;

		// use squared smooth for Gaussian
		smooth = smooth * smooth;

		float center = size / 2;
		float3 vox = make_float3(i - center, j - center, k - center);

		float dist = sqrt(vox.x * vox.x + vox.y * vox.y + vox.z * vox.z);
		float scf = (dist - rUp) * (dist - rUp);
		smooth > 0 ? scf = exp(-scf / smooth) : scf = 0;

		if (dist > rUp) {
			temp.x *= scf;
			temp.y *= scf;
		}

		scf = (dist - rDown) * (dist - rDown);
		smooth > 0 ? scf = exp(-scf / smooth) : scf = 0;

		if (dist < rDown) {
			temp.x *= scf;
			temp.y *= scf;
		}
	}
	vol[z * size * (size / 2 + 1) + y * (size / 2 + 1) + x] = temp;
}

extern "C" __global__ void MeanFree(int size, float* mask, float* vol, float2* outvol, float sum, float sum_mask){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;	
	int z = blockIdx.z * blockDim.z + threadIdx.z;	
	
	outvol[z * size * size + y * size + x] = make_float2(vol[z * size * size + y * size + x] - mask[z * size * size + y * size + x]*sum/sum_mask, 0.f);
}

extern "C" __global__ void MeanFree_RC(int size, float* mask, float* outvol, float sum, float sum_mask){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;	
	int z = blockIdx.z * blockDim.z + threadIdx.z;	
	
	outvol[z * size * size + y * size + x] = outvol[z * size * size + y * size + x] - mask[z * size * size + y * size + x]*sum/sum_mask;
}

extern "C" __global__ void correlConvConv_RC(int size, float2* partVol, float2* refVol, float2* maskVol, float2* partsqrVol)
{
        /*
         * Minor change in the first result cc new for Lasse example changed from 0.311 to 0.312
         */

        const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
        const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

        float2 ref = refVol[z * size * (size/2+1) + y * (size/2+1) + x];
        float2 par = partVol[z * size * (size/2+1) + y * (size/2+1) + x];

        float2 erg;
        erg.x = ((ref.x * par.x) + (ref.y * par.y));
        erg.y = ((ref.x * par.y) - (ref.y * par.x));

        float2 i2 = maskVol[z * size * (size/2+1) + y * (size/2+1) + x];
        float2 erg2;
        float2 erg3;
        float2 parsqr = partsqrVol[z * size * (size/2+1) + y * (size/2+1) + x];
        erg2.x = ((par.x * i2.x) - (par.y * i2.y));
        erg3.x = ((parsqr.x * i2.x) - (parsqr.y * i2.y));
        erg2.y = ((par.x * i2.y) + (par.y * i2.x));
        erg3.y = ((parsqr.x * i2.y) + (parsqr.y * i2.x));

        partVol[z * size * (size/2+1) + y * (size/2+1) + x] = erg2;
        partsqrVol[z * size * (size/2+1) + y * (size/2+1) + x] = erg3;
        refVol[z * size * (size/2+1) + y * (size/2+1) + x] = erg;
}

extern "C" __global__ void FindMax(float* maxVals, float* index, float* val, float rphi, float rpsi, float rthe){
	float oldmax = maxVals[0];
	if (val[0] > oldmax)
	{
		maxVals[0] = val[0];
		maxVals[1] = index[0];
		maxVals[2] = rphi;
		maxVals[3] = rpsi;
		maxVals[4] = rthe;
	}
}

extern "C" __global__ void MakeReal(int size, float2* inVol, float* outVol) {
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;	
	const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;	
	
	float2 temp = inVol[z * size * size + y * size + x];
	outVol[z * size * size + y * size + x] = temp.x;
}

extern "C" __global__ void MakeCplxWithSub(int size, float* inVol, float2* outVol, float val) {
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;	
	const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;	
	
	float2 temp = make_float2(inVol[z * size * size + y * size + x] - val, 0);
	outVol[z * size * size + y * size + x] = temp;
}

extern "C" __global__ void MakeCplxSqrWithSub(int size, float* inVol, float2* outVol, float val){
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;	
	const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;	
	
	float2 temp = make_float2((inVol[z * size * size + y * size + x] - val) * (inVol[z * size * size + y * size + x] - val), 0);
	outVol[z * size * size + y * size + x] = temp;
}