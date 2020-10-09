//#define USE_MPI
#ifdef USE_MPI
#include <mpi.h>
#endif
#include "default.h"
#include "Projection.h"
#include "Volume.h"
//#include "Kernels.h"
#include "hip/CudaArrays.h"
#include "hip/CudaContext.h"
#include "hip/CudaTextures.h"
#include "hip/CudaKernel.h"
#include "hip/CudaDeviceProperties.h"
#include "utils/Config.h"
//#include "utils/CudaConfig.h"
#include "utils/Matrix.h"
#include "io/Dm4FileStack.h"
#include "io/MRCFile.h"
#ifdef USE_MPI
#include "io/MPISource.h"
#endif
#include "io/MarkerFile.h"
#include "io/writeBMP.h"
#include "io/mrcHeader.h"
#include "io/emHeader.h"
#include "io/CtfFile.h"
#include "io/MotiveListe.h"
#include "io/ShiftFile.h"
#include <time.h>
#include <npp.h>
//#include "CudaKernelBinarys.h"
#include <algorithm>
#include "utils/SimpleLogger.h"
#include "Reconstructor.h"

using namespace std;
using namespace Cuda;

#ifdef WIN32
#define round(x) ((x)>=0)?(int)((x)+0.5):(int)((x)-0.5)
//#define CUDACONFFILE "cuda.cfg"
#define CONFFILE "emsart.cfg"
#else
//#define CUDACONFFILE "/home/Group/Software/tomography/kunzFunctions/EmSART/cuda.cfg"
#define CONFFILE "emsart.cfg"
#include <unistd.h>
#include <limits.h>
#endif


void WaitForInput(int exitCode)
{
	char c;
	cout << ("\nPress <Enter> to exit...");
	c = cin.get();
	exit(exitCode);
}

int main(int argc, char* argv[])
{
	int mpi_part = 0;

	int mpi_size = 1;
	const int mpi_max_name_size = 256;
	char mpi_name[mpi_max_name_size];
	int mpi_sizename = mpi_max_name_size;
	int mpi_host_id = 0;
	int mpi_host_rank = 0;
	int mpi_offset = 0;

#ifdef USE_MPI
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_part);
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
	MPI_Get_processor_name(mpi_name, &mpi_sizename);


	vector<string> hostnames;
	vector<string> singlehostnames;
	//printf("MPI process %d of %d on PC %s\n", mpi_part, mpi_size, mpi_name);

	if (mpi_part == 0)
	{
		hostnames.push_back(string(mpi_name));
		for (int i = 1; i < mpi_size; i++)
		{
			char tempname[mpi_max_name_size];
			MPI_Recv(tempname, mpi_max_name_size, MPI_CHAR, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			hostnames.push_back(string(tempname));
		}

		//printf("Found %d hostnames\n", hostnames.size());

		for (int i = 0; i < mpi_size; i++)
		{
			bool exists = false;
			for (int h = 0; h < singlehostnames.size(); h++)
			{
				if (hostnames[i] == singlehostnames[h])
					exists = true;
			}
			if (!exists)
				singlehostnames.push_back(hostnames[i]);
		}

		//sort host names alphabetically to obtain deterministic host IDs
		sort(singlehostnames.begin(), singlehostnames.end());

		for (int i = 1; i < mpi_size; i++)
		{
			int host_id;
			int host_rank = 0;
			int offset = 0;

			string hostname = hostnames[i];

			for (int h = 0; h < singlehostnames.size(); h++)
			{
				if (singlehostnames[h] == hostname)
				{
					host_id = h;
					break;
				}
			}

			for (int h = 0; h < i; h++)
			{
				if (hostnames[h] == hostname)
				{
					host_rank++;
				}
			}

			for (int h = 0; h < host_id; h++)
			{
				for (int n = 0; n < hostnames.size(); n++)
				{
					if (hostnames[n] == singlehostnames[h])
					{
						offset++;
					}
				}
			}

			MPI_Send(&host_id, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
			MPI_Send(&host_rank, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
			MPI_Send(&offset, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
		}

		for (int h = 0; h < singlehostnames.size(); h++)
		{
			if (singlehostnames[h] == string(mpi_name))
			{
				mpi_host_id = h;
				break;
			}
		}


		for (int h = 0; h < mpi_host_id; h++)
		{
			for (int n = 0; n < hostnames.size(); n++)
			{
				if (hostnames[n] == singlehostnames[h])
				{
					mpi_offset++;
				}
			}
		}
		mpi_host_rank = 0;

	}
	else
	{
		MPI_Send(mpi_name, mpi_max_name_size, MPI_CHAR, 0, 0, MPI_COMM_WORLD);

		MPI_Recv(&mpi_host_id, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(&mpi_host_rank, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(&mpi_offset, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}

	printf("Host ID: %d; host rank: %d; offset: %d; global rank: %d; name: %s\n", mpi_host_id, mpi_host_rank, mpi_offset, mpi_part, mpi_name); fflush(stdout);

	MPI_Barrier(MPI_COMM_WORLD);
#endif

	clock_t start, stop;
	double runtime = 0.0;
	CudaContext* cuCtx;

	string logfile;
	bool doLog = false;
	if (mpi_part == 0)
	{
		for (int arg = 0; arg < argc - 1; arg++)
		{
			if (string(argv[arg]) == "-log")
			{
				logfile = string(argv[arg + 1]);
				doLog = true;
			}
		}
	}

	SimpleLogger log(logfile, SimpleLogger::LOG_ERROR, !doLog);

	try
	{
		if (mpi_part == 0) printf("\n\n                          EmSART for Sub-Volumes\n\n\n");
		if (mpi_part == 0) printf("Read configuration file ");
		//Load configuration files
		Configuration::Config aConfig = Configuration::Config::GetConfig(CONFFILE, argc, argv, mpi_part, NULL);
		if (mpi_part == 0) printf("Done\n"); fflush(stdout);

		if (mpi_part == 0) printf("Projection source: %s\n", aConfig.ProjectionFile.c_str());
		if (mpi_part == 0) printf("Marker source: %s\n", aConfig.MarkerFile.c_str());
		if (mpi_part == 0) printf("Volume shifts: %f, %f, %f\n", aConfig.VolumeShift.x, aConfig.VolumeShift.y, aConfig.VolumeShift.z);
		if (mpi_part == 0) printf("Volume file name: %s\n", aConfig.OutVolumeFile.c_str());
		if (mpi_part == 0) printf("Lambda: %f\n", aConfig.Lambda);
		if (mpi_part == 0) printf("Iterations: %i\n\n", aConfig.Iterations);

#ifdef USE_MPI
		log << "Running on " << mpi_size << " GPUs in " << (int)singlehostnames.size() << " Hosts:" << endl;
		for (int i = 0; i < singlehostnames.size(); i++)
		{
			log << "Host " << i << ": " << singlehostnames[i] << endl;
		}
#else
		log << "Running in single GPU (no MPI) mode" << endl;
#endif

		log << "Configuration file: " << aConfig.GetConfigFileName() << endl;
		log << "Projection source: " << aConfig.ProjectionFile << endl;
		log << "Marker source: " << aConfig.MarkerFile << endl;
		log << "Volume file name: " << aConfig.OutVolumeFile << endl;
		log << "Volume shifts: " << aConfig.VolumeShift << endl;
		log << "Lambda: " << aConfig.Lambda << endl;
		log << "Iterations: " << aConfig.Iterations << endl;
		log << "Performing CTF correction: " << (aConfig.CtfMode != Configuration::Config::CTFM_NO ? "TRUE" : "FALSE") << endl;
		if (aConfig.CtfMode != Configuration::Config::CTFM_NO)
		{
			log << "Ignore volume Z-shift for CTF correction: " << (aConfig.IgnoreZShiftForCTF ? "TRUE" : "FALSE") << endl;
			log << "Slice thickness for CTF correction in nm: " << aConfig.CTFSliceThickness << endl;
		}

		CtfFile* defocus = NULL;

		if (aConfig.CtfMode == Configuration::Config::CTFM_YES)
		{
			defocus = new CtfFile(aConfig.CtfFile);
		}


		//Check volume dimensions:
		bool recDimOK = true;
		if (aConfig.RecDimensions.x % 4 != 0)
		{
			printf("Error: RecDimensions.x (%d) is not a multiple of 4\n", aConfig.RecDimensions.x);
			recDimOK = false;

			log << SimpleLogger::LOG_ERROR;
			log << "RecDimensions.x (" << aConfig.RecDimensions.x << ") is not a multiple of 4" << endl;
		}
		if (aConfig.RecDimensions.y % 2 != 0)
		{
			printf("Error: RecDimensions.y (%d) is not even\n", aConfig.RecDimensions.y);
			recDimOK = false;

			log << SimpleLogger::LOG_ERROR;
			log << "RecDimensions.y (" << aConfig.RecDimensions.y << ") is not even" << endl;
		}

		if (!recDimOK) WaitForInput(-1);

		printf("Create CUDA context on device %i ... \n", aConfig.CudaDeviceIDs[mpi_offset + mpi_host_rank]); fflush(stdout);
		//Create CUDA context
		cuCtx = Cuda::CudaContext::CreateInstance(aConfig.CudaDeviceIDs[mpi_offset + mpi_host_rank]);

		printf("Using CUDA device %s\n", cuCtx->GetDeviceProperties()->GetDeviceName().c_str()); fflush(stdout);

		printf("Available Memory on device: %i MB\n", cuCtx->GetFreeMemorySize() / 1024 / 1024); fflush(stdout);

		ProjectionSource* projSource;
		//Load projection data file
		if (mpi_part == 0)
		{
			if (aConfig.GetFileReadMode() == Configuration::Config::FRM_DM4)
			{
				projSource = new Dm4FileStack(aConfig.ProjectionFile);
				printf("\nLoading projections...\n");
				if (!projSource->OpenAndRead())
				{
					printf("Error: cannot read projections from %s.\n", aConfig.ProjectionFile.c_str());
					WaitForInput(-1);
				}
				projSource->ReadHeaderInfo();

				printf("Loaded %d dm4 projections.\n\n", projSource->DimZ);
			}
			else if (aConfig.GetFileReadMode() == Configuration::Config::FRM_MRC)
			{
				//Load projection data file
				projSource = new MRCFile(aConfig.ProjectionFile);
				((MRCFile*)projSource)->OpenAndReadHeader();
			}
			else
			{
				printf("Error: Projection file format not supported. Supported formats are: DM4 file series, MRC stacks, ST stacks.");
				log << SimpleLogger::LOG_ERROR;
				log << "Projection file format not supported. Supported formats are: DM4 file series, MRC stacks, ST stacks." << endl;
				WaitForInput(-1);
			}

#ifdef USE_MPI
			float pixelsize = projSource->PixelSize[0];
			int dims[4];
			dims[0] = projSource->DimX;
			dims[1] = projSource->DimY;
			dims[2] = projSource->DimZ;
			dims[3] = *((int*)&pixelsize);
			MPI_Bcast(dims, 4, MPI_INT, 0, MPI_COMM_WORLD);
#endif
		}
#ifdef USE_MPI
		else
		{
			int dims[4];
			MPI_Bcast(dims, 4, MPI_INT, 0, MPI_COMM_WORLD);
			projSource = new MPISource(dims[0], dims[1], dims[2], *((float*)&(dims[3])));
		}
#endif

		//Load marker/alignment file
		MarkerFile markers(aConfig.MarkerFile, aConfig.ReferenceMarker);

		//Create projection object to handle projection data
		Projection proj(projSource, &markers);

		MotiveList ml(aConfig.MotiveList, aConfig.ScaleMotivelistPosition, aConfig.ScaleMotivelistShift);
		
		EMFile reconstructedVol(aConfig.OutVolumeFile);
		reconstructedVol.OpenAndReadHeader();
		reconstructedVol.ReadHeaderInfo();
		dim3 volDims = make_dim3(reconstructedVol.DimX, reconstructedVol.DimY, reconstructedVol.DimZ);

		//Create volume dataset (host)
		Volume<float> *volSubVol = NULL; //this is a subVol filled with zeros to reset the storage on GPU
		Volume<float> *volSubVolReconstructed = NULL; //this storage space to contain the final data on host before saving
		vector<Volume<float>*> volSubVols; //this is an empty container to contain the localisation parameters per subVol in a batch
		Volume<float> *volReconstructed = new Volume<float>(volDims, mpi_size, -1); //empty container to get the global psoition information
		volReconstructed->PositionInSpace(aConfig.VoxelSize, aConfig.VolumeShift);
#ifdef USE_MPI
		if (aConfig.FP16Volume)
		{
			printf("FP16 volume are not supported in this version!\n");
			exit(-1);
		}
		else
		{
			//if (mpi_part == 0)
			{
				volSubVol = new Volume<float>(make_dim3(aConfig.SizeSubVol, aConfig.SizeSubVol, aConfig.SizeSubVol));
				volSubVolReconstructed = new Volume<float>(make_dim3(aConfig.SizeSubVol, aConfig.SizeSubVol, aConfig.SizeSubVol));
			}
		}
#else
		
		{
			volSubVol = new Volume<float>(make_dim3(aConfig.SizeSubVol, aConfig.SizeSubVol, aConfig.SizeSubVol));
			volSubVolReconstructed = new Volume<float>(make_dim3(aConfig.SizeSubVol, aConfig.SizeSubVol, aConfig.SizeSubVol));
		}
#endif
		


		if (aConfig.FP16Volume && !aConfig.WriteVolumeAsFP16)
			log << "; Convert to FP32 when saving to file";
		log << endl;

		float3 subVolDim;
		subVolDim = volSubVol->GetSubVolumeDimension(0);

		size_t sizeDataType;
		sizeDataType = sizeof(float);
		
		if (mpi_part == 0) printf("Memory space required by volume data: %i MB\n", aConfig.RecDimensions.x * aConfig.RecDimensions.y * aConfig.RecDimensions.z * sizeDataType / 1024 / 1024);
		if (mpi_part == 0) printf("Memory space required by partial volume: %i MB\n", aConfig.RecDimensions.x * aConfig.RecDimensions.y * (size_t)subVolDim.z * sizeDataType / 1024 / 1024);

		//Load Kernels
		KernelModuls modules(cuCtx);

		//Alloc device variables
		float3 volSize;
		CUarray_format arrayFormat;
		
		volSize = volReconstructed->GetSubVolumeDimension(mpi_part);
		arrayFormat = CU_AD_FORMAT_FLOAT;
		

		//CudaArray3D vol_Array(arrayFormat, volSize.x, volSize.y, volSize.z, 1, 2);
		//CudaTextureObject3D texObj(CU_TR_ADDRESS_MODE_CLAMP, CU_TR_ADDRESS_MODE_CLAMP, CU_TR_ADDRESS_MODE_CLAMP, CU_TR_FILTER_MODE_LINEAR, 0, &vol_Array);


		vector<CudaArray3D*> vol_ArraySubVols;
		for (size_t i = 0; i < aConfig.BatchSize; i+= mpi_size)
		{
			CudaArray3D* arr = new CudaArray3D(arrayFormat, aConfig.SizeSubVol, aConfig.SizeSubVol, aConfig.SizeSubVol, 1, 2);
			Volume<float>* v = new Volume<float>(make_dim3(aConfig.SizeSubVol, aConfig.SizeSubVol, aConfig.SizeSubVol), 1, -1);

			for (size_t m = 0; m < mpi_size; m++)
			{
				if (mpi_part == m)
				{
					vol_ArraySubVols.push_back(arr);
					volSubVols.push_back(v);
				}
				else
				{
					vol_ArraySubVols.push_back(NULL);
					volSubVols.push_back(NULL);
				}
			}
		}

		//CudaArray3D vol_ArraySubVol(arrayFormat, aConfig.SizeSubVol, aConfig.SizeSubVol, aConfig.SizeSubVol, 1, 2);
		//CudaTextureObject3D texObjSubVol(CU_TR_ADDRESS_MODE_CLAMP, CU_TR_ADDRESS_MODE_CLAMP, CU_TR_ADDRESS_MODE_CLAMP, CU_TR_FILTER_MODE_LINEAR, 0, &vol_ArraySubVol);

		CUsurfref surfref;
		cudaSafeCall(cuModuleGetSurfRef(&surfref, modules.modBP, "surfref"));
		

		
		bool volumeIsEmpty = true;

		
		int* indexList;
		int projCount;
		proj.CreateProjectionIndexList(PLT_NORMAL, &projCount, &indexList);
		//proj.CreateProjectionIndexList(PLT_RANDOM, &projCount, &indexList);
		//proj.CreateProjectionIndexList(PLT_NORMAL, &projCount, &indexList);


		if (mpi_part == 0)
		{
			printf("Projection index list:\n");
			log << "Projection index list:" << endl;
			for (uint i = 0; i < projCount; i++)
			{
				printf("%3d,", indexList[i]);
				log << indexList[i];
				if (i < projCount - 1)
					log << ", ";
			}
			log << endl;
			printf("\b \n\n");

		}

		Reconstructor reconstructor(aConfig, proj, projSource, markers, *defocus, modules, mpi_part, mpi_size);


		if (mpi_part == 0) printf("Free Memory on device after allocations: %i MB\n", cuCtx->GetFreeMemorySize() / 1024 / 1024);
		/////////////////////////////////////
		/// Filter Projections
		/////////////////////////////////////
		if (mpi_part == 0)
		{
			float lp = aConfig.fourFilterLP, hp = aConfig.fourFilterHP, lps = aConfig.fourFilterLPS, hps = aConfig.fourFilterHPS;
			bool skipFilter = aConfig.SkipFilter;

			if (!reconstructor.ComputeFourFilter() && !skipFilter)
			{
				log << SimpleLogger::LOG_ERROR;
				log << "Invalid filter parameters: Skiping filter." << endl;
				printf("Invalid filter parameters. Skiping filter...\n");
				log << SimpleLogger::LOG_INFO;
				skipFilter = true;
			}

			log << "Bandpass filter for projections applied: " << (skipFilter ? "false" : "true") << endl;
			log << "Bandpass filter values (lp, lps, hp, hps): " << lp << ", " << lps << ", " << hp << ", " << hps << endl;


			log << "Projection datatype: " << projSource->GetDataType() << endl;

			if (aConfig.ProjectionNormalization == Configuration::Config::PNM_STANDARD_DEV)
				log << "Normalizing projections by standard deviation [im = (im - mean) / std]" << endl;
			else
				log << "Normalizing projections by mean [im = (im - mean) / mean]" << endl;

			log << "Scaling projection values by: " << aConfig.ProjectionScaleFactor << endl;
			log << "Pixel size is: " << proj.GetPixelSize(0) << " nm" << endl;

			log << "Projection statistics:" << endl;

			printf("\r\n");
			for (int i = 0; i < projSource->DimZ; i++)
			{
				if (!markers.CheckIfProjIndexIsGood(i))
				{
					continue;
				}

				printf("\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b");
				printf("Filtering projection: %i", i);
				log << "Projection " << i;
				fflush(stdout);

				//projSource->GetProjection(i) always points to an array with an element size of 4 bytes,
				//Even if original data is stored in shorts! We can therfor cast data and keep the same pointer.
				char* imgUS = projSource->GetProjection(i);
				float tilt = projSource->TiltAlpha[i];
				float weight = 1.0f / cos(tilt / 180.0f * M_PI);

				//Check if data format is supported
				if (projSource->GetDataType() != FDT_SHORT &&
					projSource->GetDataType() != FDT_USHORT &&
					projSource->GetDataType() != FDT_INT &&
					projSource->GetDataType() != FDT_UINT &&
					projSource->GetDataType() != FDT_FLOAT)
				{
					cerr << "Projections have wrong data type: supported types are: short, ushort, int, uint and float.";
					log << SimpleLogger::LOG_ERROR;
					log << "Projections have wrong data type: supported types are: short, ushort, int, uint and float." << endl;
					WaitForInput(-1);
				}

				float meanValue, stdValue;
				int badPixels;
				reconstructor.PrepareProjection(imgUS, i, meanValue, stdValue, badPixels);

				printf(" Bad Pixels: %d Mean: %f Std: %f", badPixels, meanValue, stdValue);
				log << ": Bad Pixels: " << badPixels << " Mean: " << meanValue << " Std. dev.: " << stdValue << endl;
			}
		}


		/////////////////////////////////////
		/// End Filter Projections
		/////////////////////////////////////


		if (mpi_part == 0)printf("\nPixel size is: %f nm, Cs: %.2f mm, Voltage: %.2f kV\n", proj.GetPixelSize(0), aConfig.Cs, aConfig.Voltage);

		int SIRTcount = aConfig.SIRTCount;
		if (aConfig.WBP_NoSART)
			SIRTcount = 1;
		if (aConfig.WBP_NoSART)
			aConfig.Iterations = 1;


		float** SIRTBuffer = new float*[SIRTcount];
		for (int i = 0; i < SIRTcount; i++)
		{
			uint size = proj.GetWidth() * proj.GetHeight();
			SIRTBuffer[i] = new float[size];
			memset(SIRTBuffer[i], 0, size * 4);
		}

		if (mpi_part == 0)printf("\n\nStart reconstruction ...\n\n");
		fflush(stdout);
		start = clock();

		
		/*float2* extraShiftsOld = new float2[projSource->DimZ];
		int minTiltIdx = proj.GetMinimumTiltIndex();
		int minTilt = -1;
		for (size_t i = 0; i < projCount; i++)
		{
			if (indexList[i] == minTiltIdx)
			{
				minTilt = i;
				break;
			}
		}*/
		


		ShiftFile sf(aConfig.ShiftInputFile);

		float2 test = sf(1, 2);

		/*float2* extraShifts = new float2[projSource->DimZ * ml.DimY];
		memset(extraShifts, 0, projSource->DimZ * ml.DimY * sizeof(float2));

		EMFile shiftMeasured(aConfig.ShiftInputFile);
		shiftMeasured.OpenAndRead();
		shiftMeasured.ReadHeaderInfo();
		float2* s = (float2*)shiftMeasured.GetData();*/

		//invert measured shifts:
		//for (int i = 0; i < projSource->DimZ * ml.DimY; i++)
		//{
		//	s[i].x *= -1;
		//	s[i].y *= -1;
		//	//printf("Old shifts: %f, %f\n", s[i].x, s[i].y);
		//	//proj.SetExtraShift(i, s[i]);
		//}
		
		//Process particles in batches:
		for (int batch = 0; batch < ml.DimY; batch += aConfig.BatchSize)
		{
			//Reset all sub-volumes on GPU to zero:
			for (size_t i = 0; i < aConfig.BatchSize; i += mpi_size)
			{
				for (size_t m = 0; m < mpi_size; m++)
				{
					if (mpi_part == m)
					{
						vol_ArraySubVols[i + m]->CopyFromHostToArray(volSubVol->GetPtrToSubVolume(0));
					}
				}
			}

			//Loop over all projections:
			for (int i = 0; i < projCount; i++)
			{
				int index = indexList[i];
				vector<Volume<float>*> vecVols;
				vector<float2> vecExtraShifts;
				vector<CudaArray3D*> vecArrays;

				//Loop over particles in batch, batch is split over mpi nodes:
				for (int pInBatch = 0; pInBatch < aConfig.BatchSize; pInBatch += mpi_size)
				{
					int motlIdx = batch + pInBatch + mpi_part; //now we are on each node on the right index in motl! We should never see a NULL in the vectors!
					if (motlIdx >= ml.DimY)
					{
						continue; //make sure we won't pass beyond the end of the motivelist...
					}

					Volume<float>* v = volSubVols[pInBatch + mpi_part];

					motive m = ml.GetAt(motlIdx);

					float3 posSubVol = make_float3(m.x_Coord, m.y_Coord, m.z_Coord);
					float3 shift = make_float3(m.x_Shift, m.y_Shift, m.z_Shift);
					v->PositionInSpace(aConfig.VoxelSize, aConfig.VoxelSizeSubVol, *volReconstructed, posSubVol, shift);

					float2 es = sf(index, motlIdx);// s[i * projCount + motlIdx];

					float shiftLength = sqrtf(es.x * es.x + es.y * es.y);

					//printf("Extra shift for proj %d, motive %d: %f; %f\n", index, motlIdx, es.x, es.y);
					if (shiftLength < aConfig.MaxShift - 0.5f)
					{
						vecVols.push_back(v);
						vecExtraShifts.push_back(es);
						vecArrays.push_back(vol_ArraySubVols[pInBatch + mpi_part]);
					}
					//bind surfref to correct array:
					//cudaSafeCall(cuSurfRefSetArray(surfref, vol_ArraySubVols[pInBatch + mpi_part]->GetCUarray(), 0));

					//set additional shifts:
					//proj.SetExtraShift(i, s[i * projCount + motlIdx]);
				}

				//copy Data to nodes and GPU:
				if (mpi_part == 0)
				{
					//Do WBP: spread filtered projection
					memcpy(SIRTBuffer[0], projSource->GetProjection(index), (size_t)proj.GetWidth() * (size_t)proj.GetHeight() * sizeof(float));
				}

				//As compare step or file loading in WBP happens only on node 0, spread the content to all other nodes:
				reconstructor.MPIBroadcast(SIRTBuffer, 1);
				reconstructor.CopyProjectionToDevice(SIRTBuffer[0]);

				//Do Backprojection on vector data:
				reconstructor.BackProjection(volReconstructed, vecVols, vecExtraShifts, vecArrays, surfref, index);
			}

			//Write all sub-volumes to disk:
			//Loop over particles in batch, batch is split over mpi nodes:
			for (int pInBatch = 0; pInBatch < aConfig.BatchSize; pInBatch += mpi_size)
			{
				int motlIdx = batch + pInBatch + mpi_part; //now we are on each node on the right index in motl! We should never see a NULL in the vectors!
				if (motlIdx >= ml.DimY)
				{
					continue; //make sure we won't pass beyond the end of the motivelist...
				}

				motive m = ml.GetAt(motlIdx);

				vol_ArraySubVols[pInBatch + mpi_part]->CopyFromArrayToHost(volSubVolReconstructed->GetPtrToSubVolume(0));

				string filename = aConfig.SubVolPath;

				filename += m.GetIndexCoding(aConfig.NamingConv) + ".em";
				volSubVolReconstructed->Invert();
				emwrite(filename, volSubVolReconstructed->GetPtrToSubVolume(0), aConfig.SizeSubVol, aConfig.SizeSubVol, aConfig.SizeSubVol);

			}
			
		}


		/*if (!(aConfig.FP16Volume && !aConfig.WriteVolumeAsFP16))
		{
			if (mpi_part == 0) printf("\n\nCopying Data back to host ... "); fflush(stdout);

			vol_ArraySubVol.CopyFromArrayToHost(volSubVolReconstructed->GetPtrToSubVolume(0));

			if (mpi_part == 0) printf("Done\n");
		}*/

		stop = clock();
		runtime = (double)(stop - start) / CLOCKS_PER_SEC;

		if (mpi_part == 0) printf("\n\nTotal time for reconstruction: %.2i:%.2i min.\n\n", (int)floor(runtime / 60.0), (int)floor(((runtime / 60.0) - floor(runtime / 60.0))*60.0));

		
	}
	catch (exception& e)
	{
		log << SimpleLogger::LOG_ERROR;
		log << "An error occured: " << string(e.what()) << endl;
		cout << "\n\nERROR:\n";
		cout << e.what() << endl << endl;
		WaitForInput(-1);
	}
	if (mpi_part == mpi_size - 1)
		cout << endl;

	CudaContext::DestroyContext(cuCtx);
#ifdef USE_MPI
	MPI_Finalize();
#endif
}
