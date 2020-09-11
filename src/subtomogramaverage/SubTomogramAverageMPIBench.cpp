////////////////////////////////////////////////////////////////////////////////
//
//  Alexander Schröter Masterthesis Adaptation of the Original Source by
//  Michael Kunz for the Artiatomi Software Package.
//
//  Currently hosted at: https://github.com/uermel/Artiatomi
//
//  Original Source:
//
//  Copyright (c) 2018, Michael Kunz and Frangakis Lab, BMLS,
//  Goethe University, Frankfurt am Main.
//  All rights reserved.
//  http://kunzmi.github.io/Artiatomi
//
//  This file is part of the Artiatomi package.
//
//  Artiatomi is free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//
//  Artiatomi is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with Artiatomi. If not, see <http://www.gnu.org/licenses/>.
//
////////////////////////////////////////////////////////////////////////

/* AS Verbose Level
 * Setting the compiler directive to VERBOSE X will set the VERBOSE level to:
 * 0 = NO Output
 * 5 = Basic Output
 * 10 = Verbose Output
 * */
#ifndef VERBOSE
#define VERBOSE 5
#endif

/* AS
Setting the compiler directive to TIME X will add Timings to specific sections
relevant for performance measurement:
0 = No Timings
...
*/
#ifndef TIME
#define TIME 0
#endif

#define USE_MPI
#ifdef USE_MPI
#include <mpi.h>
#endif

#include <algorithm>
#include <argp.h>
#include <iomanip>
#include <map>
// #include <omp.h>
#include <argp.h>

//#include "default.h"
#include "../config/Config.h"
#include "../io/EMFile.h"
#include "../io/MotiveListe.h"

#include "../HelperFunctions.h"

#include "../hip/HipBasicKernel.h"
#include "../hip/HipContext.h"
#include "../hip/HipKernel.h"
#include "../hip/HipReducer.h"
#include "../hip/HipVariables.h"
#include "Kernels.h"

// AS HIP FFT Library is not very stable (28.2.2020) so we need to switch
// between cufft and rocfft depending on compile platform
#if defined(__HIP_PLATFORM_NVCC__) && !defined(__HIP_PLATFORM_HCC__)
#include <cufft.h>
#endif
#if !defined(__HIP_PLATFORM_NVCC__) && defined(__HIP_PLATFORM_HCC__)
#include <hipfft.h>
#endif

// AS At compile time we replace the kernels depending on which platform we
// are using. *.nvcc.h files are created using compileHIPKernelsAMD.sh and
// compileHIPKernelsNVIDIA.sh scripts
#if defined(__HIP_PLATFORM_NVCC__) && !defined(__HIP_PLATFORM_HCC__)
#include "../hip_kernels/build/basicKernels.nvcc.h"
#include "../hip_kernels/build/kernel.nvcc.h"
#endif
#if !defined(__HIP_PLATFORM_NVCC__) && defined(__HIP_PLATFORM_HCC__)
#include "../hip_kernels/build/basicKernels.hcc.h"
#include "../hip_kernels/build/kernel.hcc.h"
#endif

#include "AvgProcess.h"

#if defined(__HIP_PLATFORM_NVCC__) && !defined(__HIP_PLATFORM_HCC__)
#define grid dim3(size / 32, size/16, size)
#define block dim3(32, 16, 1)
#define grid_RC dim3((size / 2 + 1) / (size / 2 + 1), size, size)
#define block_RC dim3(size / 2 + 1, 1, 1)
#endif

#if defined(__HIP_PLATFORM_HCC__) && !defined(__HIP_PLATFORM_NVCC__)
// #define grid dim3(1, 1, size)
// #define block dim3(64, 64, 1)
// #define grid_RC dim3(size/2+1, 1, 1)
// #define block_RC dim3(1, 64, 64)

// #define grid dim3(size/64, size, size)
// #define block dim3(64, 1, 1)
// #define grid_RC dim3(size/2+1, 1, size)
// #define block_RC dim3(1, 64, 1)

#define grid dim3(size / 64, size / 4, size)
#define block dim3(64, 4, 1)
#define grid_RC dim3(size / 2 + 1, size / 64, size / 4)
#define block_RC dim3(1, 64, 4)
#endif

using namespace std;
using namespace Hip;

//#define round(x) (x >= 0 ? (int)(x + 0.5) : (int)(x - 0.5)) // TODO deprecated

extern "C" __global__ void mul(int size, float in, float2 *outVol) {
  const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

  float2 temp = outVol[z * size * size + y * size + x];
  temp.x *= in;
  temp.y *= in;
  outVol[z * size * size + y * size + x] = temp;
}

extern "C" __global__ void mulVol(int size, float *inVol, float2 *outVol) {
  const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

  float2 temp = outVol[z * size * size + y * size + x];
  temp.x *= inVol[z * size * size + y * size + x];
  temp.y *= inVol[z * size * size + y * size + x];
  outVol[z * size * size + y * size + x] = temp;
}

extern "C" __global__ void makeCplxWithSub(int size, float *inVol,
                                           float2 *outVol, float val) {
  const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

  float2 temp = make_float2(inVol[z * size * size + y * size + x] - val, 0);
  outVol[z * size * size + y * size + x] = temp;
}

extern "C" __global__ void makeReal(int size, float2 *inVol, float *outVol) {
  const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

  float2 temp = inVol[z * size * size + y * size + x];
  outVol[z * size * size + y * size + x] = temp.x;
}

extern "C" __global__ void wedgeNorm(int size, float *wedge, float2 *part,
                                     float *maxVal, int newMethod) {
  const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

  float val = wedge[z * size * size + y * size + x];

  if (newMethod) {
    if (val <= 0)
      val = 0;
    else
      val = 1.0f / val;
  } else {
    if (val < 0.1f * maxVal[0])
      val = 1.0f / (0.1f * maxVal[0]);
    else
      val = 1.0f / val;
  }
  float2 p = part[z * size * size + y * size + x];
  p.x *= val;
  p.y *= val;
  part[z * size * size + y * size + x] = p;
}

extern "C" __global__ void fftshift2(int size, float2 *volIn, float2 *volOut) {
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

extern "C" __global__ void add(int size, float *inVol, float *outVol) {
  const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

  outVol[z * size * size + y * size + x] +=
      inVol[z * size * size + y * size + x];
}

KernelModuls::KernelModuls(Hip::HipContext *aHipCtx)
    : compilerOutput(false), infoOutput(false) {
  modbasicKernels =
      aHipCtx->LoadModulePTX(KernelbasicKernels, 0, infoOutput, compilerOutput);
  modkernel =
      aHipCtx->LoadModulePTX(Kernelkernel, 0, infoOutput, compilerOutput);
}

void help() {
  cout << endl;
  // cout << "Usage: " << argv[0] << endl;
  cout << "    The following optional options override the configuration file:"
       << endl;
  cout << "    Options: " << endl;
  cout << "    -u FILENAME:   Use a user defined configuration file." << endl;
  cout << "    -h:            Show this text." << endl;

  cout << ("\nPress <Enter> to exit...");

  char c = cin.get();
  exit(-1);
}

int main(int argc, char *argv[]) {
  /* SubTomogramAveraging distributes all particles among all MPI nodes. Does
   * the Averageing for them and collects the results back on MPI node 0.
   * */

#if TIME > 0 // Setting up multiple Clocks for Timings - ToDo Put where used
  clock_t t_init_start, t_init_end, t_average_start, t_average_end,
      t_multi_ref_init_start, t_multi_ref_init_end,
      t_multi_part_iteration_start, t_multi_part_iteration_end,
      t_multi_ref_avg_start, t_multi_ref_avg_end, t_part_start, t_part_end,
      t_merge_start, t_merge_end, t_add_particles_start, t_add_particles_end,
      t_add_particles_ref_start, t_add_particles_ref_end,
      t_add_particles_ref_part_start, t_add_particles_ref_part_end,
      t_exection_time_start, t_exection_time_end;
#endif
	clock_t init_start, init_end, gpu_start, gpu_end, init2_start, init2_end, dist_start, dist_end, align_start, align_end, avg_start, avg_end, motl_start, motl_end;
	clock_t gpu_sum = 0;

  init_start = clock();

#if TIME >= 1
  t_init_start = clock();
#endif

  /* AS ***********************************************************************/
  /****************************************************************************/
  /*															  			  */
  /*         				Message Parsing Interface
   */
  /*																		  */
  /****************************************************************************/
  /****************************************************************************/

  // default values
  int mpi_part = 0;
  int mpi_size = 1;
  bool onlySumUp = false;
  const int mpi_max_name_size = 256;
  char mpi_name[mpi_max_name_size];
  int mpi_sizename = mpi_max_name_size;
  int mpi_host_id = 0;
  int mpi_host_rank = 0;
  int mpi_offset = 0;

  size_t version = atoi(argv[3]);
  size_t size = atoi(argv[4]);
  size_t maxiter = atoi(argv[5]);

  printf("Version=%i\tsize=%i\titerations=%i\n", version, size, maxiter);
  for (int i = 1; i < argc; i++) {
    string temp(argv[i]);
    if (temp == "-sumup") {
      onlySumUp = true;
    }
  }

  init_end = clock();

#ifdef USE_MPI
  dist_start = clock();
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_part);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  MPI_Get_processor_name(mpi_name, &mpi_sizename);

  vector<string> hostnames;
  vector<string> singlehostnames;
#if VERBOSE >= 1
  printf("MPI process %d of %d on PC %s\n", mpi_part, mpi_size, mpi_name);
#endif

  if (mpi_part == 0) {
    hostnames.push_back(string(mpi_name));
    for (int i = 1; i < mpi_size; i++) {
      char tempname[mpi_max_name_size];
      MPI_Recv(tempname, mpi_max_name_size, MPI_CHAR, i, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      hostnames.push_back(string(tempname));
    }

    // printf("Found %d hostnames\n", hostnames.size());

    for (int i = 0; i < mpi_size; i++) {
      bool exists = false;
      for (int h = 0; h < singlehostnames.size(); h++) {
        if (hostnames[i] == singlehostnames[h]) {
          exists = true;
        }
      }
      if (!exists) {
        singlehostnames.push_back(hostnames[i]);
      }
    }

    // sort host names alphabetically to obtain deterministic host IDs
    sort(singlehostnames.begin(), singlehostnames.end());

    for (int i = 1; i < mpi_size; i++) {
      int host_id;
      int host_rank = 0;
      int offset = 0;

      string hostname = hostnames[i];

      for (int h = 0; h < singlehostnames.size(); h++) {
        if (singlehostnames[h] == hostname) {
          host_id = h;
          break;
        }
      }

      for (int h = 0; h < i; h++) {
        if (hostnames[h] == hostname) {
          host_rank++;
        }
      }

      for (int h = 0; h < host_id; h++) {
        for (int n = 0; n < hostnames.size(); n++) {
          if (hostnames[n] == singlehostnames[h]) {
            offset++;
          }
        }
      }

      MPI_Send(&host_id, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
      MPI_Send(&host_rank, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
      MPI_Send(&offset, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
    }

    for (int h = 0; h < singlehostnames.size(); h++) {
      if (singlehostnames[h] == string(mpi_name)) {
        mpi_host_id = h;
        break;
      }
    }

    for (int h = 0; h < mpi_host_id; h++) {
      for (int n = 0; n < hostnames.size(); n++) {
        if (hostnames[n] == singlehostnames[h]) {
          mpi_offset++;
        }
      }
    }
    mpi_host_rank = 0;
  } else {
    MPI_Send(mpi_name, mpi_max_name_size, MPI_CHAR, 0, 0, MPI_COMM_WORLD);

    MPI_Recv(&mpi_host_id, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(&mpi_host_rank, 1, MPI_INT, 0, 0, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
    MPI_Recv(&mpi_offset, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

#if VERBOSE >= 1
  printf("Host ID: %d; host rank: %d; offset: %d; global rank: %d; name: %s\n",
         mpi_host_id, mpi_host_rank, mpi_offset, mpi_part, mpi_name);
  fflush(stdout);
#endif

  MPI_Barrier(MPI_COMM_WORLD);
#endif

dist_end = clock();

init2_start = clock();
  /* AS ***********************************************************************/
  /****************************************************************************/
  /*															  			  */
  /* 		        				User Input
   */
  /*																		  */
  /****************************************************************************/
  /****************************************************************************/

  int c;

  /* Flag set by ‘--verbose’. */
  static int verbose_flag = 0;
  static int c2c_flag = 1;
  string config_path = "average.cfg";

  while (true) {
    static struct option long_options[] = {
        /* These options set a flag. */
        {"verbose", no_argument, &verbose_flag, 1},
        {"brief", no_argument, &verbose_flag, 0},
        /* These options don’t set a flag. We distinguish them by their indices.
         */
        {"config", required_argument, nullptr, 'u'},
        {"C2C", no_argument, &c2c_flag, '1'},
        {"R2C", no_argument, &c2c_flag, '0'},
        {"time", required_argument, nullptr, 't'},
        {nullptr, no_argument, nullptr, 0}};
    /* getopt_long stores the option index here. */
    int option_index = 0;

    c = getopt_long(argc, argv, "u:t:", long_options, &option_index);

    /* Detect the end of the options. */
    if (c == -1)
      break;

    switch (c) {
    case 'u':
      config_path = optarg;
      printf("Configuration file is located at %s.\n", optarg);
      break;

    case 't':
      printf("The time measurement level has been set to `%i'\n", atoi(optarg));
      break;

    case '?':
      /* getopt_long already printed an error message. */
      break;

    default:
      help();
    }
  }

  /* AS ***********************************************************************/
  /****************************************************************************/
  /*															  			  */
  /*		         				Configuration							  */
  /*																		  */
  /****************************************************************************/
  /****************************************************************************/

  Configuration::Config aConfig =
      Configuration::Config::GetConfig(config_path, argc, argv, mpi_part, NULL);
  Hip::HipContext *ctx = Hip::HipContext::CreateInstance(
      aConfig.CudaDeviceIDs[mpi_part], hipDeviceScheduleSpin);


  printf("Using HIP device %s\n", ctx->GetDeviceProperties()->GetDeviceName());
  fflush(stdout);
  printf("Available Memory on device: %lu MB\n",
         ctx->GetFreeMemorySize() / 1024 / 1024);
  fflush(stdout);

  // AS Hardcode the GPU to be used when some is using GPU0 on Testsystem
  // ctx->CreateInstance(1);

  KernelModuls modules = KernelModuls(ctx);

  init2_end = clock();
#if TIME >= 1
  t_init_end = clock();
  printf("\nInitialization of Application took %f seconds.\n",
         (double)(t_init_end - t_init_start) / CLOCKS_PER_SEC);
#endif

  /* AS ***********************************************************************/
  /****************************************************************************/
  /*															  			  */
  /*      		     				Main Program							  */
  /*																		  */
  /****************************************************************************/
  /****************************************************************************/

  clock_t t_start, t_end; 


  /* Create Random Data to Average */

  srand((unsigned)time(NULL));

  float* mask_data = (float*)malloc(sizeof(float)*size*size*size);
  for (int i = 0; i < size*size*size; i++){
    mask_data[i] = (float)rand()/RAND_MAX;
  }

  float* maskcc_data = (float*)malloc(sizeof(float)*size*size*size);
  for (int i = 0; i < size*size*size; i++){
    maskcc_data[i] = (float)rand()/RAND_MAX;
  }

  float* multiref_data = (float*)malloc(sizeof(float)*size*size*size);
  for (int i = 0; i < size*size*size; i++){
    multiref_data[i] = (float)rand()/RAND_MAX;
  }

  float* filter_data = (float*)malloc(sizeof(float)*size*size*size);
  for (int i = 0; i < size*size*size; i++){
    filter_data[i] = (float)rand()/RAND_MAX;
  }

  float* wedge_data = (float*)malloc(sizeof(float)*size*size*size);
  for (int i = 0; i < size*size*size; i++){
    wedge_data[i] = (float)rand()/RAND_MAX;
  }

  float* part_data = (float*)malloc(sizeof(float)*size*size*size);
  for (int i = 0; i < size*size*size; i++){
    part_data[i] = (float)rand()/RAND_MAX;
  }

  stringstream ssml;
  ssml << aConfig.Path << aConfig.MotiveList << 1 << ".em";
  MotiveList motl(ssml.str());

  stringstream ssmlStart;
  ssmlStart << aConfig.Path << aConfig.MotiveList << aConfig.ClearAnglesIteration << ".em";
  MotiveList* motlStart = NULL;

  if (aConfig.ClearAngles){
    motlStart = new MotiveList(ssmlStart.str());
  }

  int totalCount = motl.DimY;
  int partCount = motl.DimY / mpi_size;
  int partCountArray = partCount;
  int lastPartCount = totalCount - (partCount * (mpi_size - 1));
  int startParticle = mpi_part * partCount;

  //adjust last part to fit really all particles (rounding errors...)
  if (mpi_part == mpi_size - 1){
    partCount = lastPartCount;
  }

  int endParticle = startParticle + partCount;

  if (aConfig.ClearAngles){
    for (int i = startParticle; i < endParticle; i++){
      motive m = motl.GetAt(i);
      motive mStart = motlStart->GetAt(i);

      m.phi = mStart.phi;
      m.psi = mStart.psi;
      m.theta = mStart.theta;

      motl.SetAt(i, m);
    }
  }

  stringstream ssref;
  ssref << aConfig.Path << aConfig.Reference[0] << 1 << ".em";
  EMFile ref(ssref.str());
  map<int, EMFile*> wedges;

  if (aConfig.WedgeIndices.size() < 1){
    wedges.insert(pair<int, EMFile*>(0, new EMFile(aConfig.WedgeFile)));
    wedges[0]->OpenAndRead();
    wedges[0]->ReadHeaderInfo();
  } else {
    for (size_t i = 0; i < aConfig.WedgeIndices.size(); i++){
      stringstream sswedge;
      sswedge << aConfig.WedgeFile << aConfig.WedgeIndices[i] << ".em";
      wedges.insert(pair<int, EMFile*>(aConfig.WedgeIndices[i], new EMFile(sswedge.str())));
      wedges[aConfig.WedgeIndices[i]]->OpenAndRead();
      wedges[aConfig.WedgeIndices[i]]->ReadHeaderInfo();
    }			
  }

  //EMFile wedge(aConfig.WedgeList);
  
  
  ref.OpenAndRead();
  ref.ReadHeaderInfo();
  #if DEBUG >= 1
  if (mpi_part == 0)
    cout << "ref OK" << endl;
  #endif

  EMFile mask(aConfig.Mask);
  mask.OpenAndRead();
  mask.ReadHeaderInfo();
  #if DEBUG >= 1
  if (mpi_part == 0)
    cout << "mask OK" << endl;
  #endif

  EMFile ccmask(aConfig.MaskCC);
  ccmask.OpenAndRead();
  ccmask.ReadHeaderInfo();
  #if DEBUG >= 1
  if (mpi_part == 0)
    cout << "maskcc OK" << endl;
  #endif


  EMFile* filter = NULL;
  if (aConfig.UseFilterVolume){
    filter = new EMFile(aConfig.FilterFileName);
    filter->OpenAndRead();
    filter->ReadHeaderInfo();

    #if DEBUG >= 1
    if (mpi_part == 0)
      cout << "filter OK" << endl;
    #endif
  }
  

  ////////////////////////////////////
  /// Run Average on motl fragment ///
  ////////////////////////////////////

  #if DEBUG >= 1
  std::cout << "Context OK" << std::endl;
  #endif

  //int size = ref.DimX;
  int particleSize;

  motive mot = motl.GetAt(0);
  stringstream ss;
  ss << aConfig.Path << aConfig.Particles;

  //ss << mot.partNr << ".em";
  ss << mot.GetIndexCoding(aConfig.NamingConv) << ".em";
  EMFile part(ss.str());
  part.OpenAndRead();
  part.ReadHeaderInfo();
  particleSize = part.DimX;

  AvgProcess *p;
  if (version==1){
    p = new AvgProcessC2C(
        size, 0, ctx, (float *)(mask.GetData()), (float *)(ref.GetData()),
        (float *)(ccmask.GetData()), aConfig.PhiAngIter, aConfig.PhiAngIncr,
        aConfig.AngIter, aConfig.AngIncr, aConfig.BinarizeMask,
        aConfig.RotateMaskCC, aConfig.UseFilterVolume,
        aConfig.LinearInterpolation, modules);
        //std::cout << "HIP Quality Improved Alignment using C2C (no performance improvements)" << std::endl;
  } else if (version==2) {
    p = new AvgProcessR2C(
        size, 0, ctx, (float *)(mask.GetData()), (float *)(ref.GetData()),
        (float *)(ccmask.GetData()), aConfig.PhiAngIter, aConfig.PhiAngIncr,
        aConfig.AngIter, aConfig.AngIncr, aConfig.BinarizeMask,
        aConfig.RotateMaskCC, aConfig.UseFilterVolume,
        aConfig.LinearInterpolation, modules);
  //       std::cout << "HIP Quality and Performance Improved Alignment using R2C (all improvements)" << std::endl;
  } else if (version==3) {
    p = new AvgProcessR2C_Stream(
        size, 0, ctx, (float *)(mask.GetData()), (float *)(ref.GetData()),
        (float *)(ccmask.GetData()), aConfig.PhiAngIter, aConfig.PhiAngIncr,
        aConfig.AngIter, aConfig.AngIncr, aConfig.BinarizeMask,
        aConfig.RotateMaskCC, aConfig.UseFilterVolume,
        aConfig.LinearInterpolation, modules);
        //std::cout << "unfinished improvement" << std::endl;
  } else if (version==9) {
    p = new AvgProcessOriginalBinaryKernels(
        size, 0, ctx, (float *)(mask.GetData()), (float *)(ref.GetData()),
        (float *)(ccmask.GetData()), aConfig.PhiAngIter, aConfig.PhiAngIncr,
        aConfig.AngIter, aConfig.AngIncr, aConfig.BinarizeMask,
        aConfig.RotateMaskCC, aConfig.UseFilterVolume,
        aConfig.LinearInterpolation, modules);
        //std::cout << "Original Cuda Migrated to HIP with Binary Kernels" << std::endl;
  } else {
    p = new AvgProcessOriginal(
        size, 0, ctx, (float *)(mask.GetData()), (float *)(ref.GetData()),
        (float *)(ccmask.GetData()), aConfig.PhiAngIter, aConfig.PhiAngIncr,
        aConfig.AngIter, aConfig.AngIncr, aConfig.BinarizeMask,
        aConfig.RotateMaskCC, aConfig.UseFilterVolume,
        aConfig.LinearInterpolation, modules);
      //std::cout << "HIP Original C2C (no quality or performance improvements)" << std::endl;
  }


  clock_t t_total = 0;
  hipEvent_t start, stop;
  hipEventCreate(&start);
  hipEventCreate(&stop);
  float milliseconds = 0;

  for (int iter = 0; iter < maxiter; iter++){

    srand((unsigned)time(NULL));
    for (int i = 0; i < size*size*size; i++){
      mask_data[i] = (float)rand()/RAND_MAX;
    }
    for (int i = 0; i < size*size*size; i++){
      maskcc_data[i] = (float)rand()/RAND_MAX;
    }
    for (int i = 0; i < size*size*size; i++){
      multiref_data[i] = (float)rand()/RAND_MAX;
    }
    for (int i = 0; i < size*size*size; i++){
      filter_data[i] = (float)rand()/RAND_MAX;
    }
    for (int i = 0; i < size*size*size; i++){
      wedge_data[i] = (float)rand()/RAND_MAX;
    }
    for (int i = 0; i < size*size*size; i++){
      part_data[i] = (float)rand()/RAND_MAX;
    }

    //int oldIndex = maxVals_t::getIndex(size, (int)mot.x_Shift, (int)mot.y_Shift, (int)mot.z_Shift);		
    int oldIndex = 0;

    t_start = clock();


    hipEventRecord(start);

    maxVals_t v = p->execute((float*)part_data, (float*)wedge_data, (float*)filter_data, mot.phi, mot.psi, mot.theta, (float)aConfig.HighPass, (float)aConfig.LowPass, (float)aConfig.Sigma, make_float3(mot.x_Shift, mot.y_Shift, mot.z_Shift), aConfig.CouplePhiToPsi, aConfig.ComputeCCValOnly, oldIndex);
    ctx->Synchronize();
    hipEventRecord(stop);
    hipEventSynchronize(stop);

    t_end = clock();

    t_total += t_end - t_start;
    hipEventElapsedTime(&milliseconds, start, stop);

    printf("Iterations: %i\tSize: %i\tSeconds CPU: %f\tSeconds GPU:%f\n",iter , size, (double) (t_end - t_start) / CLOCKS_PER_SEC, milliseconds);

  }

  ctx->Synchronize();
  delete p;
      
  hipDeviceReset();

  t_total = 0;


  // for (int iteration = 0; iteration < iterations; iteration++){

  //   //int oldIndex = maxVals_t::getIndex(size, (int)mot.x_Shift, (int)mot.y_Shift, (int)mot.z_Shift);		
  //   int oldIndex = 0;

  //   t_start = clock();

  //   maxVals_t v = p->execute((float*)part_data, (float*)wedge_data, (float*)filter_data, mot.phi, mot.psi, mot.theta, (float)aConfig.HighPass, (float)aConfig.LowPass, (float)aConfig.Sigma, make_float3(mot.x_Shift, mot.y_Shift, mot.z_Shift), aConfig.CouplePhiToPsi, aConfig.ComputeCCValOnly, oldIndex);

  //   t_end = clock();

  //   t_total += t_end - t_start;
  // }
  
  // ctx->Synchronize();
  // delete(p);
  // printf("Iterations: %i\tSize: %i\tSeconds: %f\tVersion: R2C\n",iter , size, (double) (t_total) / CLOCKS_PER_SEC);
  
    // hipDeviceReset();

	MPI_Finalize();
	return 0;
}

void computeRotMat(float phi, float psi, float theta, float rotMat[3][3]) {
  /* Using the 3 angles (phi, psi, theta) a rotation matrix is created.
   * AS This rotationmatrix is missing the axis definitions which are
   * probably defined by some convention.
   *
   * AS Todo I cannot imagine that this is the most efficient way to calculate
   * the rotationmatrix
   */
  int i, j;
  float sinphi, sinpsi, sintheta; /* sin of rotation angles */
  float cosphi, cospsi, costheta; /* cos of rotation angles */

  float angles[] = {0,   30,  45,  60,  90,  120, 135, 150,
                    180, 210, 225, 240, 270, 300, 315, 330};
  float angle_cos[16];
  float angle_sin[16];

  /* ToDo Values dont need to be calculated every time the computeRotMat
   * function is call */
  angle_cos[0] = 1.0f;
  angle_cos[1] = sqrt(3.0f) / 2.0f;
  angle_cos[2] = sqrt(2.0f) / 2.0f;
  angle_cos[3] = 0.5f;
  angle_cos[4] = 0.0f;
  angle_cos[5] = -0.5f;
  angle_cos[6] = -sqrt(2.0f) / 2.0f;
  angle_cos[7] = -sqrt(3.0f) / 2.0f;
  angle_cos[8] = -1.0f;
  angle_cos[9] = -sqrt(3.0f) / 2.0f;
  angle_cos[10] = -sqrt(2.0f) / 2.0f;
  angle_cos[11] = -0.5f;
  angle_cos[12] = 0.0f;
  angle_cos[13] = 0.5f;
  angle_cos[14] = sqrt(2.0f) / 2.0f;
  angle_cos[15] = sqrt(3.0f) / 2.0f;
  angle_sin[0] = 0.0f;
  angle_sin[1] = 0.5f;
  angle_sin[2] = sqrt(2.0f) / 2.0f;
  angle_sin[3] = sqrt(3.0f) / 2.0f;
  angle_sin[4] = 1.0f;
  angle_sin[5] = sqrt(3.0f) / 2.0f;
  angle_sin[6] = sqrt(2.0f) / 2.0f;
  angle_sin[7] = 0.5f;
  angle_sin[8] = 0.0f;
  angle_sin[9] = -0.5f;
  angle_sin[10] = -sqrt(2.0f) / 2.0f;
  angle_sin[11] = -sqrt(3.0f) / 2.0f;
  angle_sin[12] = -1.0f;
  angle_sin[13] = -sqrt(3.0f) / 2.0f;
  angle_sin[14] = -sqrt(2.0f) / 2.0f;
  angle_sin[15] = -0.5f;

  for (i = 0, j = 0; i < 16; i++) {
    if (angles[i] == phi) {
      cosphi = angle_cos[i];
      sinphi = angle_sin[i];
      j = 1;
    }
  }

  if (j < 1) {
    phi = phi * (float)M_PI / 180.0f;
    cosphi = cos(phi);
    sinphi = sin(phi);
  }

  for (i = 0, j = 0; i < 16; i++) {
    if (angles[i] == psi) {
      cospsi = angle_cos[i];
      sinpsi = angle_sin[i];
      j = 1;
    }
  }

  if (j < 1) {
    psi = psi * (float)M_PI / 180.0f;
    cospsi = cos(psi);
    sinpsi = sin(psi);
  }

  for (i = 0, j = 0; i < 16; i++) {
    if (angles[i] == theta) {
      costheta = angle_cos[i];
      sintheta = angle_sin[i];
      j = 1;
    }
  }

  if (j < 1) {
    theta = theta * (float)M_PI / 180.0f;
    costheta = cos(theta);
    sintheta = sin(theta);
  }

  /* ToDo This can be partial results can be reused */
  /* calculation of rotation matrix */
  rotMat[0][0] = cospsi * cosphi - costheta * sinpsi * sinphi;
  rotMat[1][0] = sinpsi * cosphi + costheta * cospsi * sinphi;
  rotMat[2][0] = sintheta * sinphi;
  rotMat[0][1] = -cospsi * sinphi - costheta * sinpsi * cosphi;
  rotMat[1][1] = -sinpsi * sinphi + costheta * cospsi * cosphi;
  rotMat[2][1] = sintheta * cosphi;
  rotMat[0][2] = sintheta * sinpsi;
  rotMat[1][2] = -sintheta * cospsi;
  rotMat[2][2] = costheta;
}

void multiplyRotMatrix(float m1[3][3], float m2[3][3], float out[3][3]) {
  /* AS ToDo Replace with Laerman Multiplication Algorithm
   *
   * https://stackoverflow.com/questions/10827209/ladermans-3x3-matrix-multiplication-with-only-23-multiplications-is-it-worth-i
   */
  out[0][0] = (float)((double)m1[0][0] * (double)m2[0][0] +
                      (double)m1[1][0] * (double)m2[0][1] +
                      (double)m1[2][0] * (double)m2[0][2]);
  out[1][0] = (float)((double)m1[0][0] * (double)m2[1][0] +
                      (double)m1[1][0] * (double)m2[1][1] +
                      (double)m1[2][0] * (double)m2[1][2]);
  out[2][0] = (float)((double)m1[0][0] * (double)m2[2][0] +
                      (double)m1[1][0] * (double)m2[2][1] +
                      (double)m1[2][0] * (double)m2[2][2]);
  out[0][1] = (float)((double)m1[0][1] * (double)m2[0][0] +
                      (double)m1[1][1] * (double)m2[0][1] +
                      (double)m1[2][1] * (double)m2[0][2]);
  out[1][1] = (float)((double)m1[0][1] * (double)m2[1][0] +
                      (double)m1[1][1] * (double)m2[1][1] +
                      (double)m1[2][1] * (double)m2[1][2]);
  out[2][1] = (float)((double)m1[0][1] * (double)m2[2][0] +
                      (double)m1[1][1] * (double)m2[2][1] +
                      (double)m1[2][1] * (double)m2[2][2]);
  out[0][2] = (float)((double)m1[0][2] * (double)m2[0][0] +
                      (double)m1[1][2] * (double)m2[0][1] +
                      (double)m1[2][2] * (double)m2[0][2]);
  out[1][2] = (float)((double)m1[0][2] * (double)m2[1][0] +
                      (double)m1[1][2] * (double)m2[1][1] +
                      (double)m1[2][2] * (double)m2[1][2]);
  out[2][2] = (float)((double)m1[0][2] * (double)m2[2][0] +
                      (double)m1[1][2] * (double)m2[2][1] +
                      (double)m1[2][2] * (double)m2[2][2]);
}

void getEulerAngles(float matrix[3][3], float &phi, float &psi, float &theta) {
  /*
   *
   */
  theta = acos(matrix[2][2]) * 180.0f / (float)M_PI;

  if (matrix[2][2] > 0.999) {
    float sign = matrix[1][0] > 0 ? 1.0f : -1.0f;
    // matrix[2][2] < 0 ? -sign : sign;
    phi = sign * acos(matrix[0][0]) * 180.0f / (float)M_PI;
    psi = 0.0f;
  } else {
    phi = atan2(matrix[2][0], matrix[2][1]) * 180.0f / (float)M_PI;
    psi = atan2(matrix[0][2], -matrix[1][2]) * 180.0f / (float)M_PI;
    // phi = atan2(matrix[0][2], matrix[2][1]) * 180.0f / (float)M_PI;
    // psi = atan2(matrix[2][0], -matrix[1][2]) * 180.0f / (float)M_PI;
  }
}

bool checkIfClassIsToAverage(vector<int> &classes, int aClass) {
  /*
   *
   *
   */
  if (classes.size() == 0) {
    return true;
  }

  for (int c : classes) {
    if (c == aClass) {
      return true;
    }
  }
  return false;
}

void printETA(size_t workitems_total, size_t workitems_done,
              clock_t time_elapsed) {
  /*
   * AS Utz wanted some sort of time left display
   * since the time needed per average doesn't change
   * we approximate by

          time elapsed / done work items * (total workitems - done work items)
   */

  printf("\tETA: %f secnonds left.", time_elapsed / workitems_done *
                                         (workitems_total - workitems_done) /
                                         CLOCKS_PER_SEC);
}
