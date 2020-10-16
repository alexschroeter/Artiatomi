////////////////////////////////////////////////////////////////////////////////
//
//  Alexander Schröter Masterthesis Migration of the Original Source by
//  Michael Kunz for the Artiatomi Software Package to HIP adding support for
//  AMD devices.
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
#define VERBOSE 1
#endif

/* AS
 * Setting the compiler directive to TIME X will add Timings to specific sections
 * relevant for performance measurement: */
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
#include <argp.h>

//#include "default.h"
#include "Config.h"
#include "EMFile.h"
#include "MotiveListe.h"
#include "../HelperFunctions.h"
#include "../hip/HipBasicKernel.h"
#include "../hip/HipContext.h"
#include "../hip/HipKernel.h"
#include "../hip/HipReducer.h"
#include "../hip/HipVariables.h"
#include "Kernels.h"

/* AS HIP FFT Library is not very stable (28.2.2020) so we need to switch
 * between cufft and rocfft depending on compile platform
 * 
 * https://github.com/ROCmSoftwarePlatform/rocFFT/issues/276
 */
#if defined(__HIP_PLATFORM_NVCC__) && !defined(__HIP_PLATFORM_HCC__)
#include <cufft.h>
#endif
#if !defined(__HIP_PLATFORM_NVCC__) && defined(__HIP_PLATFORM_HCC__)
#include <hipfft.h>
#endif

/* AS 
 * At compile time we replace the kernels depending on which platform we
 * are using. *.nvcc.h files are created using compileHIPKernelsAMD.sh and
 * compileHIPKernelsNVIDIA.sh scripts
 */
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
#define grid dim3(size / 32, size, size)
#define block dim3(32, 1, 1)
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

#define grid dim3(size / 32, size / 16, size)
#define block dim3(32, 16, 1)
#define grid_RC dim3(size / 2 + 1, size / 32, size /16)
#define block_RC dim3(1, 32, 16)
#endif

using namespace std;
using namespace Hip;

#define round(x) (x >= 0 ? (int)(x + 0.5) : (int)(x - 0.5)) // TODO deprecated

/* AS
 * In the original code most kernels had its own class which added no additional
 * functionality. This code has also been portet to hip but for ease of
 * developement in all new versions of the code we removed the classes and now 
 * simply inline the kernels. The old variant we refer to as a binary kernels 
 * since they are stored as binary. It's quite possible that this functionality 
 * wasn't available at the time of the original code but now they seem 
 * unneccessary.
 * 
 * To differentiate between Kernels in the basickernel and kernel file we call
 * kernels name those kernels needed in this file with a trailing underscore.
 * This should enable us to reduce the size in the future if we decide to remove
 * all the different kernel classes seen in the binary kernel version.
 */
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


void printETA(size_t workitems_done, size_t workitems_total, 
              clock_t time_elapsed) {
  /*
   * AS Utz wanted some sort of time left display
   * since the time needed per average doesn't change
   * we approximate by

          time elapsed / done work items * (total workitems - done work items)
   */

  //printf("\tdone %i total %i time %f\n", workitems_done, workitems_total, ((double) time_elapsed) / CLOCKS_PER_SEC);
  printf("ETA: %.0f min\n\n", (((double) time_elapsed / workitems_done * workitems_total)-(double) time_elapsed) /
                                         CLOCKS_PER_SEC / 60);
}


int main(int argc, char *argv[]) {
  /* SubTomogramAveraging distributes all particles among all MPI nodes. Does
   * the Averageing for them and collects the results back on MPI node 0.
   */

  clock_t start = clock();
  clock_t now = clock();
  /* AS
   *
   *         				Message Parsing Interface
   *
   */

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

  for (int i = 1; i < argc; i++) {
    string temp(argv[i]);
    if (temp == "-sumup") {
      onlySumUp = true;
    }
  }

#ifdef USE_MPI
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

  /* AS 
   *
   *                User Input
   * 
   */
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

  /* AS
   *
   *                          Configuration
   * 
   */
  Configuration::Config aConfig =
      Configuration::Config::GetConfig(config_path, argc, argv, mpi_part, NULL);
  Hip::HipContext *ctx = Hip::HipContext::CreateInstance(
      aConfig.CudaDeviceIDs[mpi_part], hipDeviceScheduleSpin);

  printf("GPU %i of %i is HIP device: %s with %lu MB available memory\n",mpi_part+1, mpi_size, ctx->GetDeviceProperties()->GetDeviceName(), ctx->GetFreeMemorySize() / 1024 / 1024);
  fflush(stdout);

  /* AS
   * Hardcode the GPU to be used when some is using GPU0 on Testsystem
   * ctx->CreateInstance(1);
   */

  KernelModuls modules = KernelModuls(ctx);

  /* AS
   *
   *                          Main Program
   * 
   */

  /* AS Utz wanted some sort of ETA so here we go */
  if (mpi_part == 0) {
    /* AS We measure the amount of time spent so far as a fraction of the total
     * work to be done and extrapolate the time left. We do this for
     */
  }

  size_t completed_particle = 0;
  for (int iter = aConfig.StartIteration; iter < aConfig.EndIteration; iter++) 
  {
    /*
     * beginning of the main loop
     * the config file defines how many time the routine should be performed
     */

#if VERBOSE >= 10
    printf("| Starting iteration %i of %i.\n", iter, aConfig.EndIteration);
#endif

    stringstream ssml;
    ssml << aConfig.Path << aConfig.MotiveList << iter << ".em";
    MotiveList motl(ssml.str());

#if VERBOSE >= 2
    printf("Motivelist was loaded successfully.\n");
    fflush(stdout);
#endif

    stringstream ssmlStart;
    ssmlStart << aConfig.Path << aConfig.MotiveList
              << aConfig.ClearAnglesIteration << ".em";
    MotiveList *motlStart = NULL;
    if (aConfig.ClearAngles) {
      motlStart = new MotiveList(ssmlStart.str());
    }

    int totalCount = motl.DimY;
    int partCount = motl.DimY / mpi_size;
    int partCountArray = partCount;
    int lastPartCount = totalCount - (partCount * (mpi_size - 1));
    int startParticle = mpi_part * partCount;

    // adjust last part to fit really all particles (rounding errors...)
    if (mpi_part == mpi_size - 1) {
      partCount = lastPartCount;
    }

    int endParticle = startParticle + partCount;

    if (aConfig.ClearAngles) {
#ifdef SHORTRUN
      for (int i = startParticle; i < startParticle + SHORTRUN; i++) {
#else
      for (int i = startParticle; i < endParticle; i++) {
#endif
        motive m = motl.GetAt(i);
        motive mStart = motlStart->GetAt(i);

        m.phi = mStart.phi;
        m.psi = mStart.psi;
        m.theta = mStart.theta;

        motl.SetAt(i, m);
      }
    }

    stringstream ssref;
    ssref << aConfig.Path << aConfig.Reference[0] << iter << ".em";
    EMFile ref(ssref.str());
    map<int, EMFile *> wedges;
    if (aConfig.WedgeIndices.size() < 1) {
      wedges.insert(pair<int, EMFile *>(0, new EMFile(aConfig.WedgeFile)));
      wedges[0]->OpenAndRead();
      wedges[0]->ReadHeaderInfo();
    } else {
      for (size_t i = 0; i < aConfig.WedgeIndices.size(); i++) {
        stringstream sswedge;
        sswedge << aConfig.WedgeFile << aConfig.WedgeIndices[i] << ".em";
        wedges.insert(pair<int, EMFile *>(aConfig.WedgeIndices[i],
                                          new EMFile(sswedge.str())));
        wedges[aConfig.WedgeIndices[i]]->OpenAndRead();
        wedges[aConfig.WedgeIndices[i]]->ReadHeaderInfo();
      }
    }

    // EMFile wedge(aConfig.WedgeList);
    EMFile mask(aConfig.Mask);
    EMFile ccmask(aConfig.MaskCC);
    EMFile *filter = NULL;

    ref.OpenAndRead();
    ref.ReadHeaderInfo();
#if VERBOSE >= 2
    if (mpi_part == 0)
      cout << "ref OK" << endl;
#endif

    /* AS deprecated ?
    wedge.OpenAndRead();
    wedge.ReadHeaderInfo();
    if (mpi_part == 0)
            cout << "wedge OK" << endl;
    */

    mask.OpenAndRead();
    mask.ReadHeaderInfo();
#if VERBOSE >= 2
    if (mpi_part == 0)
      cout << "mask OK" << endl;
#endif

    ccmask.OpenAndRead();
    ccmask.ReadHeaderInfo();
#if VERBOSE >= 2
    if (mpi_part == 0)
      cout << "maskcc OK" << endl;
#endif

    if (aConfig.UseFilterVolume) {
      filter = new EMFile(aConfig.FilterFileName);
      filter->OpenAndRead();
      filter->ReadHeaderInfo();
#if VERBOSE >= 2
      if (mpi_part == 0)
        cout << "filter OK" << endl;
#endif
    }

    ////////////////////////////////////
    /// Run Average on motl fragment ///
    ////////////////////////////////////

#if VERBOSE >= 2
    std::cout << "Context OK" << std::endl;
#endif

    int size = ref.DimX;
    {
      int particleSize;

      motive mot = motl.GetAt(0);
      stringstream ss;
      ss << aConfig.Path << aConfig.Particles;

      // ss << mot.partNr << ".em";
      ss << mot.GetIndexCoding(aConfig.NamingConv) << ".em";
      EMFile part(ss.str());
      part.OpenAndRead();
      part.ReadHeaderInfo();
      particleSize = part.DimX;

#if VERBOSE >= 1
      if (mpi_part == 0) {
        cout << "Checking dimensions of input data:" << endl;
        cout << "Reference: " << ref.DimX << endl;
        cout << "Particles: " << particleSize << endl;
        cout << "Wedge:     " << wedges.begin()->second->DimX << endl;
        cout << "Mask:      " << mask.DimX << endl;
        cout << "MaskCC:    " << ccmask.DimX << endl;

        if (aConfig.UseFilterVolume) {
          cout << "Filter:    " << filter->DimX << endl;
        }
      }
#endif

      if (ref.DimX != particleSize ||
          wedges.begin()->second->DimX != particleSize ||
          mask.DimX != particleSize || ccmask.DimX != particleSize) {
#if VERBOSE >= 0
        if (mpi_part == 0)
          cout << endl
               << "ERROR: not all input data dimensions are equal!" << endl;
#endif
        MPI_Finalize();
        exit(-1);
      }

      if (aConfig.UseFilterVolume) {
        if (filter->DimX != particleSize) {
#if VERBOSE >= 0
          if (mpi_part == 0)
            cout << endl
                 << "ERROR: not all input data dimensions are equal!" << endl;
#endif
          MPI_Finalize();
          exit(-1);
        }
      }
    }

#if VERBOSE >= 1
    std::cout << "Start avergaing ... (part size: " << size << ")" << std::endl;
#endif

    if (!onlySumUp)
      if (aConfig.MultiReference) {
#if VERBOSE >= 2
        printf("\t| Reference is a Multi Referencet.\n");
#endif
        
        vector<AvgProcess *> aps;
        // if (aConfig.AveragingType == "C2C"){
        //   vector<AvgProcessC2C *> aps;
        // } else if (aConfig.AveragingType == "OriginalBinary"){
        //   vector<AvgProcessOriginalBinaryKernels *> aps;
        // } else if (aConfig.AveragingType == "OriginalHIP"){
        //   vector<AvgProcessOriginal *> aps;
        // } else if (aConfig.AveragingType == "PhaseCorrelation"){
        //   vector<AvgProcessPhaseCorrelation *> aps;
        // } else {
        //   vector<AvgProcessR2C *> aps;
        // }
/* AS deprecated was used to maximize performance by chosing AveragingType at
 * compiletime
 *
#if AVGKIND == 1
        vector<AvgProcessC2C *> aps;
        std::cout << "HIP Quality Improved Alignment using C2C (no performance improvements)" << std::endl;
#elif AVGKIND == 2
        vector<AvgProcessR2C *> aps;
        std::cout << "HIP Quality and Performance Improved Alignment using R2C (all improvements)" << std::endl;
#elif AVGKIND == 3
        vector<AvgProcessR2C_Stream *> aps;
        std::cout << "unfinished Improvement" << std::endl;
#elif AVGKIND == 4
        vector<AvgProcessReal2Complex *> aps;
        std::cout << "unfinished Improvement" << std::endl;
#elif AVGKIND == 5
        vector<AvgProcessPhaseCorrelation *> aps;
        std::cout << "New Phase Correltation Method" << std::endl;
#elif AVGKIND == 9
        vector<AvgProcessOriginalBinaryKernels *> aps;
        std::cout << "Original Cuda Migrated to HIP with Binary Kernels" << std::endl;
#else
        vector<AvgProcessOriginal *> aps;
        std::cout << "HIP Original C2C (no quality or performance improvements)" << std::endl;
#endif
*/
        size_t refCount = aConfig.Reference.size();

        for (size_t ref = 0; ref < refCount; ref++) {
          stringstream ssmultiref;
          ssmultiref << aConfig.Path << aConfig.Reference[ref] << iter << ".em";
          EMFile multiref(ssmultiref.str());
          multiref.OpenAndRead();
          multiref.ReadHeaderInfo();
#if VERBOSE >= 2
          cout << "multiref OK" << endl;
#endif

        if (aConfig.AveragingType == "C2C"){
          aps.push_back(new AvgProcessC2C(
              size, 0, ctx, (float *)(mask.GetData()),
              (float *)(multiref.GetData()), (float *)(ccmask.GetData()),
              aConfig.PhiAngIter, aConfig.PhiAngIncr, aConfig.AngIter,
              aConfig.AngIncr, aConfig.BinarizeMask, aConfig.RotateMaskCC,
              aConfig.UseFilterVolume, aConfig.LinearInterpolation, modules));
        } else if (aConfig.AveragingType == "OriginalBinary"){
          aps.push_back(new AvgProcessOriginalBinaryKernels(
              size, 0, ctx, (float *)(mask.GetData()),
              (float *)(multiref.GetData()), (float *)(ccmask.GetData()),
              aConfig.PhiAngIter, aConfig.PhiAngIncr, aConfig.AngIter,
              aConfig.AngIncr, aConfig.BinarizeMask, aConfig.RotateMaskCC,
              aConfig.UseFilterVolume, aConfig.LinearInterpolation, modules));
        } else if (aConfig.AveragingType == "OriginalHIP"){
          aps.push_back(new AvgProcessOriginal(
              size, 0, ctx, (float *)(mask.GetData()),
              (float *)(multiref.GetData()), (float *)(ccmask.GetData()),
              aConfig.PhiAngIter, aConfig.PhiAngIncr, aConfig.AngIter,
              aConfig.AngIncr, aConfig.BinarizeMask, aConfig.RotateMaskCC,
              aConfig.UseFilterVolume, aConfig.LinearInterpolation, modules));
        } else if (aConfig.AveragingType == "PhaseCorrelation"){
          aps.push_back(new AvgProcessPhaseCorrelation(
              size, 0, ctx, (float *)(mask.GetData()),
              (float *)(multiref.GetData()), (float *)(ccmask.GetData()),
              aConfig.PhiAngIter, aConfig.PhiAngIncr, aConfig.AngIter,
              aConfig.AngIncr, aConfig.BinarizeMask, aConfig.RotateMaskCC,
              aConfig.UseFilterVolume, aConfig.LinearInterpolation, modules));
        } else {
            aps.push_back(new AvgProcessR2C(
              size, 0, ctx, (float *)(mask.GetData()),
              (float *)(multiref.GetData()), (float *)(ccmask.GetData()),
              aConfig.PhiAngIter, aConfig.PhiAngIncr, aConfig.AngIter,
              aConfig.AngIncr, aConfig.BinarizeMask, aConfig.RotateMaskCC,
              aConfig.UseFilterVolume, aConfig.LinearInterpolation, modules));
        }
/* AS deprecated was used to maximize performance by chosing AveragingType at
 * compiletime
 *
// #if AVGKIND == 1
//           aps.push_back(new AvgProcessC2C(
//               size, 0, ctx, (float *)(mask.GetData()),
//               (float *)(multiref.GetData()), (float *)(ccmask.GetData()),
//               aConfig.PhiAngIter, aConfig.PhiAngIncr, aConfig.AngIter,
//               aConfig.AngIncr, aConfig.BinarizeMask, aConfig.RotateMaskCC,
//               aConfig.UseFilterVolume, aConfig.LinearInterpolation, modules));
// #elif AVGKIND == 2
//           aps.push_back(new AvgProcessR2C(
//               size, 0, ctx, (float *)(mask.GetData()),
//               (float *)(multiref.GetData()), (float *)(ccmask.GetData()),
//               aConfig.PhiAngIter, aConfig.PhiAngIncr, aConfig.AngIter,
//               aConfig.AngIncr, aConfig.BinarizeMask, aConfig.RotateMaskCC,
//               aConfig.UseFilterVolume, aConfig.LinearInterpolation, modules));
// #elif AVGKIND == 3
//         aps.push_back(new AvgProcessR2C_Stream(
//             size, 0, ctx, (float *)(mask.GetData()),
//             (float *)(multiref.GetData()), (float *)(ccmask.GetData()),
//             aConfig.PhiAngIter, aConfig.PhiAngIncr, aConfig.AngIter,
//             aConfig.AngIncr, aConfig.BinarizeMask, aConfig.RotateMaskCC,
//             aConfig.UseFilterVolume, aConfig.LinearInterpolation, modules));
// #elif AVGKIND == 4
//         aps.push_back(new AvgProcessReal2Complex(
//             size, 0, ctx, (float *)(mask.GetData()),
//             (float *)(multiref.GetData()), (float *)(ccmask.GetData()),
//             aConfig.PhiAngIter, aConfig.PhiAngIncr, aConfig.AngIter,
//             aConfig.AngIncr, aConfig.BinarizeMask, aConfig.RotateMaskCC,
//             aConfig.UseFilterVolume, aConfig.LinearInterpolation, modules));
// #elif AVGKIND == 5
//         aps.push_back(new AvgProcessPhaseCorrelation(
//             size, 0, ctx, (float *)(mask.GetData()),
//             (float *)(multiref.GetData()), (float *)(ccmask.GetData()),
//             aConfig.PhiAngIter, aConfig.PhiAngIncr, aConfig.AngIter,
//             aConfig.AngIncr, aConfig.BinarizeMask, aConfig.RotateMaskCC,
//             aConfig.UseFilterVolume, aConfig.LinearInterpolation, modules));
// #elif AVGKIND == 9
//         aps.push_back(new AvgProcessOriginalBinaryKernels(
//             size, 0, ctx, (float *)(mask.GetData()),
//             (float *)(multiref.GetData()), (float *)(ccmask.GetData()),
//             aConfig.PhiAngIter, aConfig.PhiAngIncr, aConfig.AngIter,
//             aConfig.AngIncr, aConfig.BinarizeMask, aConfig.RotateMaskCC,
//             aConfig.UseFilterVolume, aConfig.LinearInterpolation, modules));
// #else
//         aps.push_back(new AvgProcessOriginal(
//             size, 0, ctx, (float *)(mask.GetData()),
//             (float *)(multiref.GetData()), (float *)(ccmask.GetData()),
//             aConfig.PhiAngIter, aConfig.PhiAngIncr, aConfig.AngIter,
//             aConfig.AngIncr, aConfig.BinarizeMask, aConfig.RotateMaskCC,
//             aConfig.UseFilterVolume, aConfig.LinearInterpolation, modules));
// #endif
*/
        }

        int motCount = partCount;

#if VERBOSE >= 2
        std::cout << "Init OK" << std::endl;
#endif

#ifdef SHORTRUN
        for (int i = startParticle; i < startParticle + SHORTRUN; i++) {
#else
        for (int i = startParticle; i < endParticle; i++) {
#endif
          motive mot = motl.GetAt(i);

          if (!checkIfClassIsToAverage(aConfig.Classes, mot.classNo)) {
            continue;
          }

          stringstream ss;
          ss << aConfig.Path << aConfig.Particles;
          ss << mot.GetIndexCoding(aConfig.NamingConv) << ".em";

#if VERBOSE >= 1
          if (mpi_part == 0)
            std::cout << "Particle: " << ss.str() << std::endl << std::flush;
#endif

          EMFile part(ss.str());
          part.OpenAndRead();
          part.ReadHeaderInfo();

#if VERBOSE >= 1
          if (mpi_part == 0)
            std::cout << "Old shift: " << mot.x_Shift << "; " << mot.y_Shift
                      << "; " << mot.z_Shift << "\tOld rot: phi: " << mot.phi << "psi: " << mot.psi
                      << "theta: " << mot.theta << std::endl;
#endif

          maxVals_t v;
          v.ccVal = -1000;

          for (int ref = 0; ref < refCount; ref++) 
          {
            int wedgeIdx = 0;
            if (aConfig.WedgeIndices.size() > 0) {
              wedgeIdx = mot.wedgeIdx;
            }
            float *filterData = NULL;
            if (aConfig.UseFilterVolume) {
              filterData = (float *)filter->GetData();
            }

            int oldIndex = maxVals_t::getIndex(
                size, (int)mot.x_Shift, (int)mot.y_Shift, (int)mot.z_Shift);

            maxVals_t temp = aps[ref]->execute(
                (float *)part.GetData(), (float *)wedges[wedgeIdx]->GetData(),
                filterData, mot.phi, mot.psi, mot.theta,
                (float)aConfig.HighPass, (float)aConfig.LowPass,
                (float)aConfig.Sigma,
                make_float3(mot.x_Shift, mot.y_Shift, mot.z_Shift),
                aConfig.CouplePhiToPsi, aConfig.ComputeCCValOnly, oldIndex);

#if VERBOSE >= 2
            cout << temp.ccVal << endl;
#endif

            if (temp.ccVal > v.ccVal) {
              v = temp;
              v.ref = ref;
            }
          }

          int sx, sy, sz;
          v.getXYZ(size, sx, sy, sz);

          float matrix1[3][3];
          float matrix2[3][3];
          float matrix3[3][3];

          computeRotMat(mot.phi, mot.psi, mot.theta, matrix1);
          computeRotMat(v.rphi, v.rpsi, v.rthe, matrix2);

          multiplyRotMatrix(matrix2, matrix1, matrix3);

          float phi, psi, theta;
          getEulerAngles(matrix3, phi, psi, theta);

          float ccOld = mot.ccCoeff;
          mot.ccCoeff = v.ccVal;
          mot.phi = phi;
          mot.psi = psi;
          mot.theta = theta;
          mot.x_Shift = (float)sx;
          mot.y_Shift = (float)sy;
          mot.z_Shift = (float)sz;
          mot.classNo = (float)v.ref + 1; // index 0 is reserved for no class!
          motl.SetAt(i, mot);

#if VERBOSE >= 1
          if (mpi_part == 0)
            cout << setprecision(0) << "I: " << (int)mot.partNr
                 << " cc:  " << fixed << setprecision(3) << v.ccVal;
          if (mpi_part == 0)
            cout.unsetf(ios_base::floatfield);
          if (mpi_part == 0)
            cout << " phi: " << setw(6) << phi << " psi: " << setw(6) << psi
                 << " the: " << setw(6) << theta << " shift: " << setw(2) << sx
                 << "," << setw(2) << sy << "," << setw(2) << sz << endl;

          if (mpi_part == 0)
            cout << setprecision(0) << "I: " << (int)mot.partNr
                 << " ref: " << fixed << setprecision(3) << v.ref + 1;
          if (mpi_part == 0)
            cout.unsetf(ios_base::floatfield);
          if (mpi_part == 0)
            cout << "     phi: " << setw(6) << v.rphi << " psi: " << setw(6)
                 << v.rpsi << " the: " << setw(6) << v.rthe << " i: " << i + 1
                 << " of " << motCount << endl;
#endif

        }
      completed_particle++;
      now = clock();
      if (mpi_part == 0){
	 printETA(completed_particle, partCount*(aConfig.EndIteration-aConfig.StartIteration), now-start);
	}
      } else {
#if VERBOSE >= 2
        printf("\t| Reference is a Single Referencet.\n");
#endif

        AvgProcess *p;
        if (aConfig.AveragingType == "C2C"){
          p = new AvgProcessC2C(
            size, 0, ctx, (float *)(mask.GetData()), (float *)(ref.GetData()),
            (float *)(ccmask.GetData()), aConfig.PhiAngIter, aConfig.PhiAngIncr,
            aConfig.AngIter, aConfig.AngIncr, aConfig.BinarizeMask,
            aConfig.RotateMaskCC, aConfig.UseFilterVolume,
            aConfig.LinearInterpolation, modules);
        } else if (aConfig.AveragingType == "OriginalBinary"){
          p = new AvgProcessOriginalBinaryKernels(
            size, 0, ctx, (float *)(mask.GetData()), (float *)(ref.GetData()),
            (float *)(ccmask.GetData()), aConfig.PhiAngIter, aConfig.PhiAngIncr,
            aConfig.AngIter, aConfig.AngIncr, aConfig.BinarizeMask,
            aConfig.RotateMaskCC, aConfig.UseFilterVolume,
            aConfig.LinearInterpolation, modules);
        } else if (aConfig.AveragingType == "OriginalHIP"){
          p = new AvgProcessOriginal(
            size, 0, ctx, (float *)(mask.GetData()), (float *)(ref.GetData()),
            (float *)(ccmask.GetData()), aConfig.PhiAngIter, aConfig.PhiAngIncr,
            aConfig.AngIter, aConfig.AngIncr, aConfig.BinarizeMask,
            aConfig.RotateMaskCC, aConfig.UseFilterVolume,
            aConfig.LinearInterpolation, modules);
        } else if (aConfig.AveragingType == "PhaseCorrelation"){
          p = new AvgProcessPhaseCorrelation(
            size, 0, ctx, (float *)(mask.GetData()), (float *)(ref.GetData()),
            (float *)(ccmask.GetData()), aConfig.PhiAngIter, aConfig.PhiAngIncr,
            aConfig.AngIter, aConfig.AngIncr, aConfig.BinarizeMask,
            aConfig.RotateMaskCC, aConfig.UseFilterVolume,
            aConfig.LinearInterpolation, modules);
        } else {
          p = new AvgProcessR2C(
            size, 0, ctx, (float *)(mask.GetData()), (float *)(ref.GetData()),
            (float *)(ccmask.GetData()), aConfig.PhiAngIter, aConfig.PhiAngIncr,
            aConfig.AngIter, aConfig.AngIncr, aConfig.BinarizeMask,
            aConfig.RotateMaskCC, aConfig.UseFilterVolume,
            aConfig.LinearInterpolation, modules);
        } 
/* AS deprecated was used to maximize performance by chosing AveragingType at
 * compiletime
 *
// #if AVGKIND == 1
//         p = new AvgProcessC2C(
//             size, 0, ctx, (float *)(mask.GetData()), (float *)(ref.GetData()),
//             (float *)(ccmask.GetData()), aConfig.PhiAngIter, aConfig.PhiAngIncr,
//             aConfig.AngIter, aConfig.AngIncr, aConfig.BinarizeMask,
//             aConfig.RotateMaskCC, aConfig.UseFilterVolume,
//             aConfig.LinearInterpolation, modules);
//             //std::cout << "HIP Quality Improved Alignment using C2C (no performance improvements)" << std::endl;
// #elif AVGKIND == 2
//         p = new AvgProcessR2C(
//             size, 0, ctx, (float *)(mask.GetData()), (float *)(ref.GetData()),
//             (float *)(ccmask.GetData()), aConfig.PhiAngIter, aConfig.PhiAngIncr,
//             aConfig.AngIter, aConfig.AngIncr, aConfig.BinarizeMask,
//             aConfig.RotateMaskCC, aConfig.UseFilterVolume,
//             aConfig.LinearInterpolation, modules);
//             //std::cout << "HIP Quality and Performance Improved Alignment using R2C (all improvements)" << std::endl;
// #elif AVGKIND == 3
//         p = new AvgProcessR2C_Stream(
//             size, 0, ctx, (float *)(mask.GetData()), (float *)(ref.GetData()),
//             (float *)(ccmask.GetData()), aConfig.PhiAngIter, aConfig.PhiAngIncr,
//             aConfig.AngIter, aConfig.AngIncr, aConfig.BinarizeMask,
//             aConfig.RotateMaskCC, aConfig.UseFilterVolume,
//             aConfig.LinearInterpolation, modules);
//             //std::cout << "unfinished improvement" << std::endl;
// #elif AVGKIND == 5
//         p = new AvgProcessPhaseCorrelation(
//             size, 0, ctx, (float *)(mask.GetData()), (float *)(ref.GetData()),
//             (float *)(ccmask.GetData()), aConfig.PhiAngIter, aConfig.PhiAngIncr,
//             aConfig.AngIter, aConfig.AngIncr, aConfig.BinarizeMask,
//             aConfig.RotateMaskCC, aConfig.UseFilterVolume,
//             aConfig.LinearInterpolation, modules);
//             //std::cout << "New Phase Correltation Method" << std::endl;
// #elif AVGKIND == 9
//         p = new AvgProcessOriginalBinaryKernels(
//             size, 0, ctx, (float *)(mask.GetData()), (float *)(ref.GetData()),
//             (float *)(ccmask.GetData()), aConfig.PhiAngIter, aConfig.PhiAngIncr,
//             aConfig.AngIter, aConfig.AngIncr, aConfig.BinarizeMask,
//             aConfig.RotateMaskCC, aConfig.UseFilterVolume,
//             aConfig.LinearInterpolation, modules);
//             //std::cout << "Original Cuda Migrated to HIP with Binary Kernels" << std::endl;
// #else
//         p = new AvgProcessOriginal(
//             size, 0, ctx, (float *)(mask.GetData()), (float *)(ref.GetData()),
//             (float *)(ccmask.GetData()), aConfig.PhiAngIter, aConfig.PhiAngIncr,
//             aConfig.AngIter, aConfig.AngIncr, aConfig.BinarizeMask,
//             aConfig.RotateMaskCC, aConfig.UseFilterVolume,
//             aConfig.LinearInterpolation, modules);
//           //std::cout << "HIP Original C2C (no quality or performance improvements)" << std::endl;
// #endif
*/
        int motCount = partCount;

#if VERBOSE >= 2
        std::cout << "Init OK" << std::endl;
#endif

#ifdef SHORTRUN
        // AS For smaller runs and tests
        for (int i = startParticle; i < startParticle + SHORTRUN; i++) {
#else
        for (int i = startParticle; i < endParticle; i++) {
#endif

          motive mot = motl.GetAt(i);

          if (!checkIfClassIsToAverage(aConfig.Classes, mot.classNo)) {
            continue;
          }

          stringstream ss;
          ss << aConfig.Path << aConfig.Particles;
          ss << mot.GetIndexCoding(aConfig.NamingConv) << ".em";

#if VERBOSE >= 1
          if (mpi_part == 0)
            std::cout << "Particle: " << ss.str() << std::endl << std::flush;
#endif

          EMFile part(ss.str());
          part.OpenAndRead();
          part.ReadHeaderInfo();

#if VERBOSE >= 1
          if (mpi_part == 0)
            std::cout << "Old shift: " << mot.x_Shift << "; " << mot.y_Shift
                      << "; " << mot.z_Shift <<  "\tOld rot: phi: " << mot.phi << " psi: " << mot.psi
                      << " theta: " << mot.theta << std::endl;
#endif

          int wedgeIdx = 0;
          if (aConfig.WedgeIndices.size() > 0) {
            wedgeIdx = mot.wedgeIdx;
          }
          float *filterData = NULL;
          if (aConfig.UseFilterVolume) {
            filterData = (float *)filter->GetData();
          }

          int oldIndex = maxVals_t::getIndex(
              size, (int)mot.x_Shift, (int)mot.y_Shift, (int)mot.z_Shift);

          maxVals_t v = p->execute(
              (float *)part.GetData(), (float *)wedges[wedgeIdx]->GetData(),
              filterData, mot.phi, mot.psi, mot.theta, (float)aConfig.HighPass,
              (float)aConfig.LowPass, (float)aConfig.Sigma,
              make_float3(mot.x_Shift, mot.y_Shift, mot.z_Shift),
              aConfig.CouplePhiToPsi, aConfig.ComputeCCValOnly, oldIndex);

          int sx, sy, sz;
          v.getXYZ(size, sx, sy, sz);

          float matrix1[3][3];
          float matrix2[3][3];
          float matrix3[3][3];

          computeRotMat(mot.phi, mot.psi, mot.theta, matrix1);
          computeRotMat(v.rphi, v.rpsi, v.rthe, matrix2);

          multiplyRotMatrix(matrix2, matrix1, matrix3);

// #if VERBOSE >= 1
//           if (mpi_part == 0){
//             cout << "Rotationmatrix Inverse\n" << endl;
//             cout << matrix3[0][0] << "\t"<< matrix3[1][0] << "\t"<< matrix3[2][0] << endl;
//             cout << matrix3[0][1] << "\t"<< matrix3[1][1] << "\t"<< matrix3[2][1] << endl;
//             cout << matrix3[0][2] << "\t"<< matrix3[1][2] << "\t"<< matrix3[2][2] << endl;
//           }
// #endif      

          float phi, psi, theta;
          getEulerAngles(matrix3, phi, psi, theta);

          float ccOld = mot.ccCoeff;
          mot.ccCoeff = v.ccVal;
          mot.phi = phi;
          mot.psi = psi;
          mot.theta = theta;
          mot.x_Shift = (float)sx;
          mot.y_Shift = (float)sy;
          mot.z_Shift = (float)sz;
          motl.SetAt(i, mot);

#if VERBOSE >= 1
          if (mpi_part == 0)
            cout << setprecision(0) << "I: " << (int)mot.partNr
                 << " cc old: " << fixed << setprecision(3) << ccOld;
          if (mpi_part == 0)
            cout.unsetf(ios_base::floatfield);
          if (mpi_part == 0)
            cout << " phi: " << setw(6) << phi << " psi: " << setw(6) << psi
                 << " the: " << setw(6) << theta << " shift: " << setw(2) << sx
                 << "," << setw(2) << sy << "," << setw(2) << sz << endl;

          if (mpi_part == 0)
            cout << setprecision(0) << "I: " << (int)mot.partNr
                 << " cc new: " << fixed << setprecision(3) << v.ccVal;
          if (mpi_part == 0)
            cout.unsetf(ios_base::floatfield);
          if (mpi_part == 0)
            cout << " phi: " << setw(6) << v.rphi << " psi: " << setw(6)
                 << v.rpsi << " the: " << setw(6) << v.rthe << " i: " << i + 1
                 << " of " << motCount << endl;
#endif
        completed_particle++;
        now = clock();
        if (mpi_part == 0){
		printETA(completed_particle, partCount*(aConfig.EndIteration-aConfig.StartIteration), now-start);
	}
        }

      }

    ctx->Synchronize();

    ///////////////////////////////////////
    /// End of Average on motl fragment ///
    ///////////////////////////////////////

    MPI_Barrier(MPI_COMM_WORLD);

    /////////////////////////////////
    /// Merge partial motivelists ///
    /////////////////////////////////

    float meanCCValue = 0;
    if (mpi_part == 0) {
      float *buffer =
          new float[motl.DimX *
                    (partCount > lastPartCount ? partCount : lastPartCount)];
      float *motlBuffer = (float *)motl.GetData();

      for (int mpi = 1; mpi < mpi_size - 1; mpi++) {
        // cout << mpi_part << ": " << partCount << endl;
        MPI_Recv(buffer, motl.DimX * partCount, MPI_FLOAT, mpi, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        memcpy(&motlBuffer[motl.DimX * mpi * partCount], buffer,
               partCount * motl.DimX * sizeof(float));
        // cout << mpi_part << ": " << buffer[0] << endl;
      }

      if (mpi_size > 1)
        for (int mpi = mpi_size - 1; mpi < mpi_size; mpi++) {
          // cout << mpi_part << ": " << lastPartCount << endl;
          MPI_Recv(buffer, motl.DimX * lastPartCount, MPI_FLOAT, mpi, 0,
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          memcpy(&motlBuffer[motl.DimX * mpi * partCount], buffer,
                 lastPartCount * motl.DimX * sizeof(float));
          // cout << mpi_part << ": " << buffer[0] << endl;
        }

      double counter = 0;
      double meanCC = 0;

      float *tempCC = new float[totalCount];

      for (size_t i = 0; i < totalCount; i++) {
        motive mot2 = motl.GetAt(i);
        meanCC += mot2.ccCoeff;
        counter++;
        tempCC[i] = mot2.ccCoeff;
      }

      sort(tempCC, tempCC + totalCount);

      size_t idx = (size_t)(totalCount * (1.0f - aConfig.BestParticleRatio));
      if (idx > totalCount - 1)
        idx = totalCount - 1;

      if (idx < 0)
        idx = 0;
      meanCCValue = tempCC[idx];
      delete[] tempCC;

      if (!onlySumUp) {
        // save motiveList
        stringstream ssmlNew;
        ssmlNew << aConfig.Path << aConfig.MotiveList << iter + 1 << ".em";
        emwrite(ssmlNew.str(), motlBuffer, motl.DimX, motl.DimY, 1);
      }

      motive *motlMot = (motive *)motlBuffer;
      for (size_t i = 0; i < totalCount; i++) {
        // mark bad particles with too low ccCoeff with a negative class number,
        // only if it isn't already negative
        if (motlMot[i].ccCoeff < meanCCValue && motlMot[i].classNo < 0) {
          motlMot[i].classNo *= -1;
        }
        // if ccCoeff got better in that iteration, remove the negative mark to
        // re-integrate the particle. This keeps the total amount of particle
        // constant!
        if (motlMot[i].ccCoeff >= meanCCValue && motlMot[i].classNo < 0) {
          motlMot[i].classNo *= -1;
        }
      }

      if (!onlySumUp) {
        // save motiveList with used particles
        stringstream ssmlUsed;
        ssmlUsed << aConfig.Path << aConfig.MotiveList << iter + 1 << ".em";
        emwrite(ssmlUsed.str(), motlBuffer, motl.DimX, motl.DimY, 1);
      }
      delete[] buffer;

      /*meanCC /= counter;
      meanCCValue = (float)meanCC;*/
      for (int mpi = 1; mpi < mpi_size; mpi++) {
        MPI_Send(&meanCCValue, 1, MPI_FLOAT, mpi, 0, MPI_COMM_WORLD);
      }
    } else {
      float *motlBuffer = (float *)motl.GetData();
      // cout << mpi_part << ": " << partCount << ": " << startParticle << endl;
      // in last part, partCount is equal to lastPartCount!
      MPI_Send(&motlBuffer[motl.DimX * mpi_part * partCountArray],
               motl.DimX * partCount, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
      MPI_Recv(&meanCCValue, 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
    }

    /////////////////////////////////
    /// End of motivelist merging ///
    /////////////////////////////////

    MPI_Barrier(MPI_COMM_WORLD);

    /////////////////////
    /// Add particles ///
    /////////////////////

    /* AS Converted Version of Add Particle */
    {

      hipModule_t modbasicKernels =
          ctx->LoadModulePTX(KernelbasicKernels, 0, false, false);
      hipModule_t modkernel = ctx->LoadModulePTX(Kernelkernel, 0, false, false);

      hipStream_t stream = 0;

#if defined(__HIP_PLATFORM_HCC__) && !defined(__HIP_PLATFORM_NVCC__)
      int n[] = {(int)size, (int)size, (int)size};
      hipfftHandle ffthandle;
      hipfftPlanMany(&ffthandle, 3, n, NULL, 0, 0, NULL, 0, 0, HIPFFT_C2C, 1);
      hipfftSetStream(ffthandle, stream);
#else
      int n[] = {(int)size, (int)size, (int)size};
      cufftHandle ffthandle;
      cufftPlanMany(&ffthandle, 3, n, NULL, 0, 0, NULL, 0, 0, CUFFT_C2C, 1);
      cufftSetStream(ffthandle, stream);
#endif

      // HipRot rot(size, stream, ctx, aConfig.LinearInterpolation);
      RotateKernel aRotateKernelROT(modbasicKernels, grid, block, size,
                                    aConfig.LinearInterpolation);
      // HipRot rotWedge(size, stream, ctx, aConfig.LinearInterpolation);
      RotateKernel aRotateKernelROTWEDGE(modbasicKernels, grid, block, size,
                                         aConfig.LinearInterpolation);

      // HipSub sub(size, stream, ctx);
      // HipMakeCplxWithSub makecplx(size, stream, ctx);
      // HipBinarize binarize(size, stream, ctx);
      // HipMul mul(size, stream, ctx);
      // HipFFT fft(size, stream, ctx);
      // HipReducer max(size*size*size, stream, ctx);
      Reducer aReduceKernel(modkernel, grid, block);
      // HipWedgeNorm wedgeNorm(size, stream, ctx);

      HipDeviceVariable partReal(size * size * size * sizeof(float));
      HipDeviceVariable partRot(size * size * size * sizeof(float));
      HipDeviceVariable partCplx(size * size * size * sizeof(float2));
      HipDeviceVariable wedge_d(size * size * size * sizeof(float));
      HipDeviceVariable wedgeSum(size * size * size * sizeof(float));
      HipDeviceVariable wedgeSumO(size * size * size * sizeof(float));
      HipDeviceVariable wedgeSumE(size * size * size * sizeof(float));
      HipDeviceVariable wedgeSumA(size * size * size * sizeof(float));
      HipDeviceVariable wedgeSumB(size * size * size * sizeof(float));
      HipDeviceVariable tempCplx(size * size * size * sizeof(float2));
      HipDeviceVariable temp(size * size * size * sizeof(float));
      HipDeviceVariable partSum(size * size * size * sizeof(float));
      HipDeviceVariable partSumEven(size * size * size * sizeof(float));
      HipDeviceVariable partSumOdd(size * size * size * sizeof(float));
      HipDeviceVariable partSumA(size * size * size * sizeof(float));
      HipDeviceVariable partSumB(size * size * size * sizeof(float));

      int skipped = 0;
      vector<int> partsPerRef;

      for (size_t ref = 0; ref < aConfig.Reference.size(); ref++) 
      {
        float currentReference = ref + 1;

        partSum.Memset(0);
        partSumOdd.Memset(0);
        partSumEven.Memset(0);
        partSumA.Memset(0);
        partSumB.Memset(0);

        wedgeSum.Memset(0);
        wedgeSumO.Memset(0);
        wedgeSumE.Memset(0);
        wedgeSumA.Memset(0);
        wedgeSumB.Memset(0);

        int sumCount = 0;
        int motCount = partCount;

        float limit = 0;

        limit = meanCCValue;
        int oldWedgeIdx = -1;

#ifdef SHORTRUN
        for (int i = startParticle; i < startParticle + SHORTRUN; i++) {
#else
        for (int i = startParticle; i < endParticle; i++) {
#endif

          motive mot = motl.GetAt(i);
          stringstream ss;
          ss << aConfig.Path << aConfig.Particles;

          if (mot.classNo != currentReference && aConfig.Reference.size() > 1) {
            continue;
          }
          if (mot.ccCoeff < limit) {
            skipped++;
            continue;
          }

          if (oldWedgeIdx != mot.wedgeIdx) {
            oldWedgeIdx = 0;
            if (aConfig.WedgeIndices.size() > 0) {
              oldWedgeIdx = mot.wedgeIdx;
            }

            wedge_d.CopyHostToDevice((float *)wedges[oldWedgeIdx]->GetData());
            aRotateKernelROTWEDGE.SetTexture(wedge_d);
          }
          sumCount++;

          ss << mot.GetIndexCoding(aConfig.NamingConv) << ".em";

#if VERBOSE >= 1
          cout << mpi_part << ": "
               << "Part nr: " << mot.partNr << " ref: " << currentReference
               << " summed up: " << sumCount << " skipped: " << skipped << " = "
               << sumCount + skipped << " of " << motCount << endl;
#endif

          EMFile part(ss.str());
          part.OpenAndRead();
          part.ReadHeaderInfo();

          int size = part.DimX;

          partReal.CopyHostToDevice(part.GetData());

          float3 shift;
          shift.x = -mot.x_Shift;
          shift.y = -mot.y_Shift;
          shift.z = -mot.z_Shift;

          aRotateKernelROT.SetTextureShift(partReal);
          /* */
          aRotateKernelROT.do_shift(size, partRot, shift);

          aRotateKernelROT.SetTexture(partRot);
          aRotateKernelROT.do_rotate_improved(size, partReal, -mot.psi, -mot.phi,
                                     -mot.theta);
          /* */
          /*
          aRotateKernelROT.do_shiftrot3d(size, partReal, -mot.psi, -mot.phi,
                                     -mot.theta, shift);
          */

          aRotateKernelROTWEDGE.do_rotate_improved(size, wedge_d, -mot.psi, -mot.phi,
                                          -mot.theta);
          // sub.Add(wedge_d, wedgeSum);
          hipLaunchKernelGGL(add, grid, block, 0, stream, size,
                             (float *)wedge_d.GetDevicePtr(),
                             (float *)wedgeSum.GetDevicePtr());

          // makecplx.MakeCplxWithSub(partReal, partCplx, 0);
          hipLaunchKernelGGL(makeCplxWithSub, grid, block, 0, stream, size,
                             (float *)partReal.GetDevicePtr(),
                             (float2 *)partCplx.GetDevicePtr(), 0);

          tempCplx.CopyDeviceToDevice(partCplx);
#if defined(__HIP_PLATFORM_HCC__)
          // hipMemcpy((hipfftComplex*)tempCplx.GetDevicePtr(),
          // (hipfftComplex*)partCplx.GetDevicePtr(),
          // sizeof(hipfftComplex)*size*size*size, hipMemcpyDeviceToDevice);
          hipfftExecC2C(ffthandle, (hipfftComplex *)tempCplx.GetDevicePtr(),
                        (hipfftComplex *)tempCplx.GetDevicePtr(),
                        HIPFFT_FORWARD);
#else
          // cudaMemcpy((cufftComplex*)tempCplx.GetDevicePtr(),
          // (cufftComplex*)partCplx.GetDevicePtr(),
          // sizeof(cufftComplex)*size*size*size, cudaMemcpyDeviceToDevice);
          cufftExecC2C(ffthandle, (cufftComplex *)tempCplx.GetDevicePtr(),
                       (cufftComplex *)tempCplx.GetDevicePtr(), CUFFT_FORWARD);
#endif

          // fft.FFTShift2(tempCplx, partCplx);
          hipLaunchKernelGGL(fftshift2, grid, block, 0, stream, size,
                             (float2 *)tempCplx.GetDevicePtr(),
                             (float2 *)partCplx.GetDevicePtr());

          // mul.MulVolCplx(wedge_d, partCplx);
          hipLaunchKernelGGL(mulVol, grid, block, 0, stream, size,
                             (float *)wedge_d.GetDevicePtr(),
                             (float2 *)partCplx.GetDevicePtr());
          /* AF Elementwise Multiplcation of Wedge and Particle */

          // fft.FFTShift2(partCplx, tempCplx);
          hipLaunchKernelGGL(fftshift2, grid, block, 0, stream, size,
                             (float2 *)partCplx.GetDevicePtr(),
                             (float2 *)tempCplx.GetDevicePtr());

          partCplx.CopyDeviceToDevice(tempCplx);
#if defined(__HIP_PLATFORM_HCC__)
          hipfftExecC2C(ffthandle, (hipfftComplex *)partCplx.GetDevicePtr(),
                        (hipfftComplex *)partCplx.GetDevicePtr(),
                        HIPFFT_BACKWARD);
#else
          cufftExecC2C(ffthandle, (cufftComplex *)partCplx.GetDevicePtr(),
                       (cufftComplex *)partCplx.GetDevicePtr(), CUFFT_INVERSE);
#endif

          // mul.Mul(1.0f / (float)size / (float)size / (float)size, partCplx);
          hipLaunchKernelGGL(mul, grid, block, 0, stream, size,
                             1.0f / (float)size / (float)size / (float)size,
                             (float2 *)partCplx.GetDevicePtr());

          // makecplx.MakeReal(partCplx, partReal);
          hipLaunchKernelGGL(makeReal, grid, block, 0, stream, size,
                             (float2 *)partCplx.GetDevicePtr(),
                             (float *)partReal.GetDevicePtr());

          // sub.Add(partReal, partSum);
          hipLaunchKernelGGL(add, grid, block, 0, stream, size,
                             (float *)partReal.GetDevicePtr(),
                             (float *)partSum.GetDevicePtr());

          if (sumCount % 2 == 0) {
            // sub.Add(partReal, partSumEven);
            hipLaunchKernelGGL(add, grid, block, 0, stream, size,
                               (float *)partReal.GetDevicePtr(),
                               (float *)partSumEven.GetDevicePtr());
            // sub.Add(wedge_d, wedgeSumE);
            hipLaunchKernelGGL(add, grid, block, 0, stream, size,
                               (float *)wedge_d.GetDevicePtr(),
                               (float *)wedgeSumE.GetDevicePtr());
          } else {
            // sub.Add(partReal, partSumOdd);
            hipLaunchKernelGGL(add, grid, block, 0, stream, size,
                               (float *)partReal.GetDevicePtr(),
                               (float *)partSumOdd.GetDevicePtr());
            // sub.Add(wedge_d, wedgeSumO);
            hipLaunchKernelGGL(add, grid, block, 0, stream, size,
                               (float *)wedge_d.GetDevicePtr(),
                               (float *)wedgeSumO.GetDevicePtr());
          }

          if (i < motCount / 2) {
            // sub.Add(partReal, partSumA);
            hipLaunchKernelGGL(add, grid, block, 0, stream, size,
                               (float *)partReal.GetDevicePtr(),
                               (float *)partSumA.GetDevicePtr());
            // sub.Add(wedge_d, wedgeSumA);
            hipLaunchKernelGGL(add, grid, block, 0, stream, size,
                               (float *)wedge_d.GetDevicePtr(),
                               (float *)wedgeSumA.GetDevicePtr());
          } else {
            // sub.Add(partReal, partSumB);
            hipLaunchKernelGGL(add, grid, block, 0, stream, size,
                               (float *)partReal.GetDevicePtr(),
                               (float *)partSumB.GetDevicePtr());
            // sub.Add(wedge_d, wedgeSumB);
            hipLaunchKernelGGL(add, grid, block, 0, stream, size,
                               (float *)wedge_d.GetDevicePtr(),
                               (float *)wedgeSumB.GetDevicePtr());
          }
        }

        partsPerRef.push_back(sumCount);

        if (mpi_part == 0) {
          float *buffer = new float[size * size * size];
          for (int mpi = 1; mpi < mpi_size; mpi++) {
            MPI_Recv(buffer, size * size * size, MPI_FLOAT, mpi, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            partReal.CopyHostToDevice(buffer);
            // sub.Add(partReal, partSum);
            hipLaunchKernelGGL(add, grid, block, 0, stream, size,
                               (float *)partReal.GetDevicePtr(),
                               (float *)partSum.GetDevicePtr());
            MPI_Recv(buffer, size * size * size, MPI_FLOAT, mpi, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            partReal.CopyHostToDevice(buffer);
            // sub.Add(partReal, wedgeSum);
            hipLaunchKernelGGL(add, grid, block, 0, stream, size,
                               (float *)partReal.GetDevicePtr(),
                               (float *)wedgeSum.GetDevicePtr());

            MPI_Recv(buffer, size * size * size, MPI_FLOAT, mpi, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            partReal.CopyHostToDevice(buffer);
            // sub.Add(partReal, partSumEven);
            hipLaunchKernelGGL(add, grid, block, 0, stream, size,
                               (float *)partReal.GetDevicePtr(),
                               (float *)partSumEven.GetDevicePtr());
            MPI_Recv(buffer, size * size * size, MPI_FLOAT, mpi, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            partReal.CopyHostToDevice(buffer);
            // sub.Add(partReal, wedgeSumE);
            hipLaunchKernelGGL(add, grid, block, 0, stream, size,
                               (float *)partReal.GetDevicePtr(),
                               (float *)wedgeSumE.GetDevicePtr());

            MPI_Recv(buffer, size * size * size, MPI_FLOAT, mpi, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            partReal.CopyHostToDevice(buffer);
            // sub.Add(partReal, partSumOdd);
            hipLaunchKernelGGL(add, grid, block, 0, stream, size,
                               (float *)partReal.GetDevicePtr(),
                               (float *)partSumOdd.GetDevicePtr());
            MPI_Recv(buffer, size * size * size, MPI_FLOAT, mpi, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            partReal.CopyHostToDevice(buffer);
            // sub.Add(partReal, wedgeSumO);
            hipLaunchKernelGGL(add, grid, block, 0, stream, size,
                               (float *)partReal.GetDevicePtr(),
                               (float *)wedgeSumO.GetDevicePtr());

            MPI_Recv(buffer, size * size * size, MPI_FLOAT, mpi, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            partReal.CopyHostToDevice(buffer);
            // sub.Add(partReal, partSumA);
            hipLaunchKernelGGL(add, grid, block, 0, stream, size,
                               (float *)partReal.GetDevicePtr(),
                               (float *)partSumA.GetDevicePtr());
            MPI_Recv(buffer, size * size * size, MPI_FLOAT, mpi, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            partReal.CopyHostToDevice(buffer);
            // sub.Add(partReal, wedgeSumA);
            hipLaunchKernelGGL(add, grid, block, 0, stream, size,
                               (float *)partReal.GetDevicePtr(),
                               (float *)wedgeSumA.GetDevicePtr());

            MPI_Recv(buffer, size * size * size, MPI_FLOAT, mpi, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            partReal.CopyHostToDevice(buffer);
            // sub.Add(partReal, partSumB);
            hipLaunchKernelGGL(add, grid, block, 0, stream, size,
                               (float *)partReal.GetDevicePtr(),
                               (float *)partSumB.GetDevicePtr());
            MPI_Recv(buffer, size * size * size, MPI_FLOAT, mpi, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            partReal.CopyHostToDevice(buffer);
            // sub.Add(partReal, wedgeSumB);
            hipLaunchKernelGGL(add, grid, block, 0, stream, size,
                               (float *)partReal.GetDevicePtr(),
                               (float *)wedgeSumB.GetDevicePtr());
          }
          delete[] buffer;
        } else {
          float *buffer = new float[size * size * size];

          partSum.CopyDeviceToHost(buffer);
          MPI_Send(buffer, size * size * size, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
          wedgeSum.CopyDeviceToHost(buffer);
          MPI_Send(buffer, size * size * size, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);

          partSumEven.CopyDeviceToHost(buffer);
          MPI_Send(buffer, size * size * size, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
          wedgeSumE.CopyDeviceToHost(buffer);
          MPI_Send(buffer, size * size * size, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);

          partSumOdd.CopyDeviceToHost(buffer);
          MPI_Send(buffer, size * size * size, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
          wedgeSumO.CopyDeviceToHost(buffer);
          MPI_Send(buffer, size * size * size, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);

          partSumA.CopyDeviceToHost(buffer);
          MPI_Send(buffer, size * size * size, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
          wedgeSumA.CopyDeviceToHost(buffer);
          MPI_Send(buffer, size * size * size, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);

          partSumB.CopyDeviceToHost(buffer);
          MPI_Send(buffer, size * size * size, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
          wedgeSumB.CopyDeviceToHost(buffer);
          MPI_Send(buffer, size * size * size, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);

          delete[] buffer;
        }

        if (mpi_part == 0) {
          // max.MaxIndex(wedgeSum, temp, tempCplx);
          aReduceKernel.maxindex(wedgeSum, temp, tempCplx, size * size * size);

          /* TODO AS Write Particle into EM file */
          /*float* testParticle = new float[size*size*size];
          partSum.CopyDeviceToHost(testParticle);
          emwrite("testParticle.em", testParticle, size, size, size);
          delete[] testParticle;
          /**/
          
          // makecplx.MakeCplxWithSub(partSum, partCplx, 0);
          hipLaunchKernelGGL(makeCplxWithSub, grid, block, 0, stream, size,
                             (float *)partSum.GetDevicePtr(),
                             (float2 *)partCplx.GetDevicePtr(), 0);

          tempCplx.CopyDeviceToDevice(partCplx);
#if defined(__HIP_PLATFORM_HCC__)
          hipfftExecC2C(ffthandle, (hipfftComplex *)tempCplx.GetDevicePtr(),
                        (hipfftComplex *)tempCplx.GetDevicePtr(),
                        HIPFFT_FORWARD);
#else
          cufftExecC2C(ffthandle, (cufftComplex *)tempCplx.GetDevicePtr(),
                       (cufftComplex *)tempCplx.GetDevicePtr(), CUFFT_FORWARD);
#endif

          // fft.FFTShift2(tempCplx, partCplx);
          hipLaunchKernelGGL(fftshift2, grid, block, 0, stream, size,
                             (float2 *)tempCplx.GetDevicePtr(),
                             (float2 *)partCplx.GetDevicePtr());

          /* TODO AS Write Wedge into EM file */
          /*float* testWedge = new float[size*size*size];
          wedgeSum.CopyDeviceToHost(testWedge);
          emwrite("testWedge.em", testWedge, size, size, size);
          delete[] testWedge;

          float maxVal = 0;
          temp.CopyDeviceToHost(&maxVal, sizeof(float));
          cout << "Max value wedge: " << maxVal << endl;
          /* */

          /* AS TODO WRITE WEDGE SUM TO EMFILE */

          // wedgeNorm.WedgeNorm(wedgeSum, partCplx, temp, 0);
          hipLaunchKernelGGL(wedgeNorm, grid, block, 0, stream, size,
                             (float *)wedgeSum.GetDevicePtr(),
                             (float2 *)partCplx.GetDevicePtr(),
                             (float *)temp.GetDevicePtr(), 1);

          // fft.FFTShift2(partCplx, tempCplx);
          hipLaunchKernelGGL(fftshift2, grid, block, 0, stream, size,
                             (float2 *)partCplx.GetDevicePtr(),
                             (float2 *)tempCplx.GetDevicePtr());

          partCplx.CopyDeviceToDevice(tempCplx);
#if defined(__HIP_PLATFORM_HCC__)
          hipfftExecC2C(ffthandle, (hipfftComplex *)partCplx.GetDevicePtr(),
                        (hipfftComplex *)partCplx.GetDevicePtr(),
                        HIPFFT_BACKWARD);
#else
          cufftExecC2C(ffthandle, (cufftComplex *)partCplx.GetDevicePtr(),
                       (cufftComplex *)partCplx.GetDevicePtr(), CUFFT_INVERSE);
#endif

          // mul.Mul(1.0f / (float)size / (float)size / (float)size, partCplx);
          hipLaunchKernelGGL(mul, grid, block, 0, stream, size,
                             1.0f / (float)size / (float)size / (float)size,
                             (float2 *)partCplx.GetDevicePtr());

          // makecplx.MakeReal(partCplx, partSum);
          hipLaunchKernelGGL(makeReal, grid, block, 0, stream, size,
                             (float2 *)partCplx.GetDevicePtr(),
                             (float *)partSum.GetDevicePtr());

          // max.MaxIndex(wedgeSumO, temp, tempCplx);
          aReduceKernel.maxindex(wedgeSumO, temp, tempCplx, size * size * size);

          // makecplx.MakeCplxWithSub(partSumOdd, partCplx, 0);
          hipLaunchKernelGGL(makeCplxWithSub, grid, block, 0, stream, size,
                             (float *)partSumOdd.GetDevicePtr(),
                             (float2 *)partCplx.GetDevicePtr(), 0);

          tempCplx.CopyDeviceToDevice(partCplx);
#if defined(__HIP_PLATFORM_HCC__)
          hipfftExecC2C(ffthandle, (hipfftComplex *)tempCplx.GetDevicePtr(),
                        (hipfftComplex *)tempCplx.GetDevicePtr(),
                        HIPFFT_FORWARD);
#else
          cufftExecC2C(ffthandle, (cufftComplex *)tempCplx.GetDevicePtr(),
                       (cufftComplex *)tempCplx.GetDevicePtr(), CUFFT_FORWARD);
#endif

          // fft.FFTShift2(tempCplx, partCplx);
          hipLaunchKernelGGL(fftshift2, grid, block, 0, stream, size,
                             (float2 *)tempCplx.GetDevicePtr(),
                             (float2 *)partCplx.GetDevicePtr());

          // wedgeNorm.WedgeNorm(wedgeSumO, partCplx, temp, 0);
          hipLaunchKernelGGL(wedgeNorm, grid, block, 0, stream, size,
                             (float *)wedgeSumO.GetDevicePtr(),
                             (float2 *)partCplx.GetDevicePtr(),
                             (float *)temp.GetDevicePtr(), 0);

          // fft.FFTShift2(partCplx, tempCplx);
          hipLaunchKernelGGL(fftshift2, grid, block, 0, stream, size,
                             (float2 *)partCplx.GetDevicePtr(),
                             (float2 *)tempCplx.GetDevicePtr());

          partCplx.CopyDeviceToDevice(tempCplx);
#if defined(__HIP_PLATFORM_HCC__)
          hipfftExecC2C(ffthandle, (hipfftComplex *)partCplx.GetDevicePtr(),
                        (hipfftComplex *)partCplx.GetDevicePtr(),
                        HIPFFT_BACKWARD);
#else
          cufftExecC2C(ffthandle, (cufftComplex *)partCplx.GetDevicePtr(),
                       (cufftComplex *)partCplx.GetDevicePtr(), CUFFT_INVERSE);
#endif

          // mul.Mul(1.0f / (float)size / (float)size / (float)size, partCplx);
          hipLaunchKernelGGL(mul, grid, block, 0, stream, size,
                             1.0f / (float)size / (float)size / (float)size,
                             (float2 *)partCplx.GetDevicePtr());

          // makecplx.MakeReal(partCplx, partSumOdd);
          hipLaunchKernelGGL(makeReal, grid, block, 0, stream, size,
                             (float2 *)partCplx.GetDevicePtr(),
                             (float *)partSumOdd.GetDevicePtr());

          // max.MaxIndex(wedgeSumE, temp, tempCplx);
          aReduceKernel.maxindex(wedgeSumE, temp, tempCplx, size * size * size);

          // makecplx.MakeCplxWithSub(partSumEven, partCplx, 0);
          hipLaunchKernelGGL(makeCplxWithSub, grid, block, 0, stream, size,
                             (float *)partSumEven.GetDevicePtr(),
                             (float2 *)partCplx.GetDevicePtr(), 0);

          tempCplx.CopyDeviceToDevice(partCplx);
#if defined(__HIP_PLATFORM_HCC__)
          hipfftExecC2C(ffthandle, (hipfftComplex *)tempCplx.GetDevicePtr(),
                        (hipfftComplex *)tempCplx.GetDevicePtr(),
                        HIPFFT_FORWARD);
#else
          cufftExecC2C(ffthandle, (cufftComplex *)tempCplx.GetDevicePtr(),
                       (cufftComplex *)tempCplx.GetDevicePtr(), CUFFT_FORWARD);
#endif

          // fft.FFTShift2(tempCplx, partCplx);
          hipLaunchKernelGGL(fftshift2, grid, block, 0, stream, size,
                             (float2 *)tempCplx.GetDevicePtr(),
                             (float2 *)partCplx.GetDevicePtr());

          // wedgeNorm.WedgeNorm(wedgeSumE, partCplx, temp, 0);
          hipLaunchKernelGGL(wedgeNorm, grid, block, 0, stream, size,
                             (float *)wedgeSumE.GetDevicePtr(),
                             (float2 *)partCplx.GetDevicePtr(),
                             (float *)temp.GetDevicePtr(), 0);

          // fft.FFTShift2(partCplx, tempCplx);
          hipLaunchKernelGGL(fftshift2, grid, block, 0, stream, size,
                             (float2 *)partCplx.GetDevicePtr(),
                             (float2 *)tempCplx.GetDevicePtr());

          partCplx.CopyDeviceToDevice(tempCplx);
#if defined(__HIP_PLATFORM_HCC__)
          hipfftExecC2C(ffthandle, (hipfftComplex *)partCplx.GetDevicePtr(),
                        (hipfftComplex *)partCplx.GetDevicePtr(),
                        HIPFFT_BACKWARD);
#else
          cufftExecC2C(ffthandle, (cufftComplex *)partCplx.GetDevicePtr(),
                       (cufftComplex *)partCplx.GetDevicePtr(), CUFFT_INVERSE);
#endif

          // mul.Mul(1.0f / (float)size / (float)size / (float)size, partCplx);
          hipLaunchKernelGGL(mul, grid, block, 0, stream, size,
                             1.0f / (float)size / (float)size / (float)size,
                             (float2 *)partCplx.GetDevicePtr());

          // makecplx.MakeReal(partCplx, partSumEven);
          hipLaunchKernelGGL(makeReal, grid, block, 0, stream, size,
                             (float2 *)partCplx.GetDevicePtr(),
                             (float *)partSumEven.GetDevicePtr());

          // max.MaxIndex(wedgeSumA, temp, tempCplx);
          aReduceKernel.maxindex(wedgeSumA, temp, tempCplx, size * size * size);

          // makecplx.MakeCplxWithSub(partSumA, partCplx, 0);
          hipLaunchKernelGGL(makeCplxWithSub, grid, block, 0, stream, size,
                             (float *)partSumA.GetDevicePtr(),
                             (float2 *)partCplx.GetDevicePtr(), 0);

          tempCplx.CopyDeviceToDevice(partCplx);
#if defined(__HIP_PLATFORM_HCC__)
          hipfftExecC2C(ffthandle, (hipfftComplex *)tempCplx.GetDevicePtr(),
                        (hipfftComplex *)tempCplx.GetDevicePtr(),
                        HIPFFT_FORWARD);
#else
          cufftExecC2C(ffthandle, (cufftComplex *)tempCplx.GetDevicePtr(),
                       (cufftComplex *)tempCplx.GetDevicePtr(), CUFFT_FORWARD);
#endif

          // fft.FFTShift2(tempCplx, partCplx);
          hipLaunchKernelGGL(fftshift2, grid, block, 0, stream, size,
                             (float2 *)tempCplx.GetDevicePtr(),
                             (float2 *)partCplx.GetDevicePtr());

          // wedgeNorm.WedgeNorm(wedgeSumA, partCplx, temp, 0);
          hipLaunchKernelGGL(wedgeNorm, grid, block, 0, stream, size,
                             (float *)wedgeSumA.GetDevicePtr(),
                             (float2 *)partCplx.GetDevicePtr(),
                             (float *)temp.GetDevicePtr(), 0);

          // fft.FFTShift2(partCplx, tempCplx);
          hipLaunchKernelGGL(fftshift2, grid, block, 0, stream, size,
                             (float2 *)partCplx.GetDevicePtr(),
                             (float2 *)tempCplx.GetDevicePtr());

          partCplx.CopyDeviceToDevice(tempCplx);
#if defined(__HIP_PLATFORM_HCC__)
          hipfftExecC2C(ffthandle, (hipfftComplex *)partCplx.GetDevicePtr(),
                        (hipfftComplex *)partCplx.GetDevicePtr(),
                        HIPFFT_BACKWARD);
#else
          cufftExecC2C(ffthandle, (cufftComplex *)partCplx.GetDevicePtr(),
                       (cufftComplex *)partCplx.GetDevicePtr(), CUFFT_INVERSE);
#endif

          // mul.Mul(1.0f / (float)size / (float)size / (float)size, partCplx);
          hipLaunchKernelGGL(mul, grid, block, 0, stream, size,
                             1.0f / (float)size / (float)size / (float)size,
                             (float2 *)partCplx.GetDevicePtr());

          // makecplx.MakeReal(partCplx, partSumA);
          hipLaunchKernelGGL(makeReal, grid, block, 0, stream, size,
                             (float2 *)partCplx.GetDevicePtr(),
                             (float *)partSumA.GetDevicePtr());

          // max.MaxIndex(wedgeSumB, temp, tempCplx);
          aReduceKernel.maxindex(wedgeSumB, temp, tempCplx, size * size * size);

          // makecplx.MakeCplxWithSub(partSumB, partCplx, 0);
          hipLaunchKernelGGL(makeCplxWithSub, grid, block, 0, stream, size,
                             (float *)partSumB.GetDevicePtr(),
                             (float2 *)partCplx.GetDevicePtr(), 0);

          tempCplx.CopyDeviceToDevice(partCplx);
#if defined(__HIP_PLATFORM_HCC__)
          hipfftExecC2C(ffthandle, (hipfftComplex *)tempCplx.GetDevicePtr(),
                        (hipfftComplex *)tempCplx.GetDevicePtr(),
                        HIPFFT_FORWARD);
#else
          cufftExecC2C(ffthandle, (cufftComplex *)tempCplx.GetDevicePtr(),
                       (cufftComplex *)tempCplx.GetDevicePtr(), CUFFT_FORWARD);
#endif

          // fft.FFTShift2(tempCplx, partCplx);
          hipLaunchKernelGGL(fftshift2, grid, block, 0, stream, size,
                             (float2 *)tempCplx.GetDevicePtr(),
                             (float2 *)partCplx.GetDevicePtr());

          // wedgeNorm.WedgeNorm(wedgeSumB, partCplx, temp, 0);
          hipLaunchKernelGGL(wedgeNorm, grid, block, 0, stream, size,
                             (float *)wedgeSumB.GetDevicePtr(),
                             (float2 *)partCplx.GetDevicePtr(),
                             (float *)temp.GetDevicePtr(), 0);

          // fft.FFTShift2(partCplx, tempCplx);
          hipLaunchKernelGGL(fftshift2, grid, block, 0, stream, size,
                             (float2 *)partCplx.GetDevicePtr(),
                             (float2 *)tempCplx.GetDevicePtr());

          partCplx.CopyDeviceToDevice(tempCplx);
#if defined(__HIP_PLATFORM_HCC__)
          hipfftExecC2C(ffthandle, (hipfftComplex *)partCplx.GetDevicePtr(),
                        (hipfftComplex *)partCplx.GetDevicePtr(),
                        HIPFFT_BACKWARD);
#else
          cufftExecC2C(ffthandle, (cufftComplex *)partCplx.GetDevicePtr(),
                       (cufftComplex *)partCplx.GetDevicePtr(), CUFFT_INVERSE);
#endif

          // mul.Mul(1.0f / (float)size / (float)size / (float)size, partCplx);
          hipLaunchKernelGGL(mul, grid, block, 0, stream, size,
                             1.0f / (float)size / (float)size / (float)size,
                             (float2 *)partCplx.GetDevicePtr());

          // makecplx.MakeReal(partCplx, partSumB);
          hipLaunchKernelGGL(makeReal, grid, block, 0, stream, size,
                             (float2 *)partCplx.GetDevicePtr(),
                             (float *)partSumB.GetDevicePtr());

          float *sum = new float[size * size * size];

          if (aConfig.ApplySymmetry == Configuration::Symmetry_Rotate180) {
            float *nowedge = new float[size * size * size];
            float *part = new float[size * size * size];
            for (size_t i = 0; i < size * size * size; i++) {
              nowedge[i] = 1;
            }
            partSum.CopyDeviceToHost(sum);
            aRotateKernelROT.SetOldAngles(0, 0, 0);
            aRotateKernelROT.SetTexture(partSum);
            aRotateKernelROT.do_rotate_improved(size, temp, 180.0f, 0, 0);
            temp.CopyDeviceToHost(part);

            /* old code
            emwrite("testpart.em", part, size, size, size);
            emwrite("testSum.em", sum, size, size, size);
            */

            AvgProcess *p;
            if (aConfig.AveragingType == "C2C"){
              p = new AvgProcessC2C(size, 0, ctx, sum, nowedge, nowedge, 1, 0, 5,
                    3, aConfig.BinarizeMask, aConfig.RotateMaskCC,
                    aConfig.UseFilterVolume,
                    aConfig.LinearInterpolation, modules);
            } else if (aConfig.AveragingType == "OriginalBinary"){
              p = new AvgProcessOriginalBinaryKernels(size, 0, ctx, sum, 
                    nowedge, nowedge, 1, 0, 5, 3, aConfig.BinarizeMask,
                    aConfig.RotateMaskCC, aConfig.UseFilterVolume,
                    aConfig.LinearInterpolation, modules);
            } else if (aConfig.AveragingType == "OriginalHIP"){
              p = new AvgProcessOriginal(size, 0, ctx, sum, nowedge, nowedge, 
                    1, 0, 5, 3, aConfig.BinarizeMask, aConfig.RotateMaskCC,
                    aConfig.UseFilterVolume,
                    aConfig.LinearInterpolation, modules);
            } else if (aConfig.AveragingType == "PhaseCorrelation"){
              p = new AvgProcessPhaseCorrelation(size, 0, ctx, sum, nowedge,
                     nowedge, 1, 0, 5, 3, aConfig.BinarizeMask,
                    aConfig.RotateMaskCC, aConfig.UseFilterVolume,
                    aConfig.LinearInterpolation, modules);
            } else {
              p = new AvgProcessR2C(size, 0, ctx, sum, nowedge, nowedge, 1, 0, 5,
                    3, aConfig.BinarizeMask, aConfig.RotateMaskCC,
                    aConfig.UseFilterVolume,
                    aConfig.LinearInterpolation, modules);
            } 
/* AS deprecated was used to maximize performance by chosing AveragingType at
 * compiletime
 *
// #if AVGKIND == 1
//       p = new AvgProcessC2C(size, 0, ctx, sum, nowedge, nowedge, 1, 0, 5,
//                                   3, aConfig.BinarizeMask, aConfig.RotateMaskCC,
//                                   aConfig.UseFilterVolume,
//                                   aConfig.LinearInterpolation, modules);
//       std::cout << "HIP Quality Improved Alignment using C2C (no performance improvements)" << std::endl;  
// #elif AVGKIND == 2
//       p = new AvgProcessR2C(size, 0, ctx, sum, nowedge, nowedge, 1, 0, 5,
//                               3, aConfig.BinarizeMask, aConfig.RotateMaskCC,
//                               aConfig.UseFilterVolume,
//                               aConfig.LinearInterpolation, modules);
//       std::cout << "HIP Quality and Performance Improved Alignment using R2C (all improvements)" << std::endl;
// #elif AVGKIND == 3
//       p = new AvgProcessR2C_Stream(
//           size, 0, ctx, sum, nowedge, nowedge, 1, 0, 5, 3, aConfig.BinarizeMask,
//           aConfig.RotateMaskCC, aConfig.UseFilterVolume,
//           aConfig.LinearInterpolation, modules);
//       std::cout << "unfinished Improvement" << std::endl;
// #elif AVGKIND == 4
//       p = new AvgProcessReal2Complex(
//           size, 0, ctx, sum, nowedge, nowedge, 1, 0, 5, 3, aConfig.BinarizeMask,
//           aConfig.RotateMaskCC, aConfig.UseFilterVolume,
//           aConfig.LinearInterpolation, modules);
//       std::cout << "unfinished Improvement" << std::endl;
// #elif AVGKIND == 5
//       p = new AvgProcessPhaseCorrelation(
//           size, 0, ctx, sum, nowedge, nowedge, 1, 0, 5, 3, aConfig.BinarizeMask,
//           aConfig.RotateMaskCC, aConfig.UseFilterVolume,
//           aConfig.LinearInterpolation, modules);
//       std::cout << "New Phase Correlation Method" << std::endl;
// #elif AVGKIND == 9
//       p = new AvgProcessOriginalBinaryKernels(
//           size, 0, ctx, sum, nowedge, nowedge, 1, 0, 5, 3, aConfig.BinarizeMask,
//           aConfig.RotateMaskCC, aConfig.UseFilterVolume,
//           aConfig.LinearInterpolation, modules);
//       std::cout << "Original Cuda Migrated to HIP with Binary Kernels" << std::endl;

// #else
//       p = new AvgProcessOriginal(size, 0, ctx, sum, nowedge, nowedge, 1, 0, 5,
//                                  3, aConfig.BinarizeMask, aConfig.RotateMaskCC,
//                                  aConfig.UseFilterVolume,
//                                  aConfig.LinearInterpolation, modules);
//       std::cout << "HIP Original C2C (no quality or performance improvements)" << std::endl;
// #endif
*/
            float *filterData = NULL;
            if (aConfig.UseFilterVolume) {
              filterData = (float *)filter->GetData();
            }
            maxVals_t v = p->execute(
                part, nowedge, filterData, 0, 0, 0, (float)aConfig.HighPass,
                (float)aConfig.LowPass, (float)aConfig.Sigma,
                make_float3(0, 0, 0), aConfig.CouplePhiToPsi, false, 0);

            int sx, sy, sz;
            v.getXYZ(size, sx, sy, sz);

#if VERBOSE >= 1
            cout << mpi_part << ": "
                 << "Found shift for symmetry: " << sx << ", " << sy << ", "
                 << sz << v.ccVal << endl;
            cout << mpi_part << ": "
                 << "Found PSI/Theta for symmetry: " << v.rphi << " / "
                 << v.rthe << " CC-Val: " << v.ccVal << endl;
#endif

            float3 shift;
            shift.x = -sx;
            shift.y = -sy;
            shift.z = -sz;

            aRotateKernelROT.SetTextureShift(temp);
            aRotateKernelROT.do_shift(size, partRot, shift);

            aRotateKernelROT.SetTexture(partRot);
            aRotateKernelROT.do_rotate_improved(size, partReal, 0, -v.rphi, 0);

            // sub.Add(partReal, partSum);
            hipLaunchKernelGGL(add, grid, block, 0, stream, size,
                               (float *)partReal.GetDevicePtr(),
                               (float *)partSum.GetDevicePtr());
            delete[] nowedge;
            delete[] part;
          }

          if (aConfig.ApplySymmetry == Configuration::Symmetry_Shift) {
            // partSum is now the averaged Particle without symmetry
            aRotateKernelROT.SetTextureShift(partSum);

            if (!(aConfig.ShiftSymmetryVector[0].x == 0 &&
                  aConfig.ShiftSymmetryVector[0].y == 0 &&
                  aConfig.ShiftSymmetryVector[0].z == 0)) {
              aRotateKernelROT.do_shift(size, partReal,
                                        aConfig.ShiftSymmetryVector[0]);
              // sub.Add(partReal, partSum);
              hipLaunchKernelGGL(add, grid, block, 0, stream, size,
                                 (float *)partReal.GetDevicePtr(),
                                 (float *)partSum.GetDevicePtr());

              aRotateKernelROT.do_shift(size, partReal,
                                        -aConfig.ShiftSymmetryVector[0]);
              // sub.Add(partReal, partSum);
              hipLaunchKernelGGL(add, grid, block, 0, stream, size,
                                 (float *)partReal.GetDevicePtr(),
                                 (float *)partSum.GetDevicePtr());
            }

            if (!(aConfig.ShiftSymmetryVector[1].x == 0 &&
                  aConfig.ShiftSymmetryVector[1].y == 0 &&
                  aConfig.ShiftSymmetryVector[1].z == 0)) {
              aRotateKernelROT.do_shift(size, partReal,
                                        aConfig.ShiftSymmetryVector[1]);
              // sub.Add(partReal, partSum);
              hipLaunchKernelGGL(add, grid, block, 0, stream, size,
                                 (float *)partReal.GetDevicePtr(),
                                 (float *)partSum.GetDevicePtr());

              aRotateKernelROT.do_shift(size, partReal,
                                        -aConfig.ShiftSymmetryVector[1]);
              // sub.Add(partReal, partSum);
              hipLaunchKernelGGL(add, grid, block, 0, stream, size,
                                 (float *)partReal.GetDevicePtr(),
                                 (float *)partSum.GetDevicePtr());
            }

            if (!(aConfig.ShiftSymmetryVector[2].x == 0 &&
                  aConfig.ShiftSymmetryVector[2].y == 0 &&
                  aConfig.ShiftSymmetryVector[2].z == 0)) {
              aRotateKernelROT.do_shift(size, partReal,
                                        aConfig.ShiftSymmetryVector[2]);
              // sub.Add(partReal, partSum);
              hipLaunchKernelGGL(add, grid, block, 0, stream, size,
                                 (float *)partReal.GetDevicePtr(),
                                 (float *)partSum.GetDevicePtr());

              aRotateKernelROT.do_shift(size, partReal,
                                        -aConfig.ShiftSymmetryVector[2]);
              // sub.Add(partReal, partSum);
              hipLaunchKernelGGL(add, grid, block, 0, stream, size,
                                 (float *)partReal.GetDevicePtr(),
                                 (float *)partSum.GetDevicePtr());
            }
          }

          if (aConfig.ApplySymmetry == Configuration::Symmetry_Helical) {
            partSum.CopyDeviceToHost(sum);
            stringstream ss1;
            string outName = aConfig.Path + aConfig.Reference[ref] + "noSymm_";
            ss1 << outName << iter + 1 << ".em";
            emwrite(ss1.str(), sum, size, size, size);

            // partSum is now the averaged Particle without symmetry
            aRotateKernelROT.SetTexture(partSum);

            /*float rise = 22.92f / (49.0f / 3.0f) / (1.1f * 2);
            float twist = 360.0f / 49.0f * 3.0f;*/
            float rise = aConfig.HelicalRise;
            float twist = aConfig.HelicalTwist;

            for (int i = aConfig.HelicalRepeatStart;
                 i <= aConfig.HelicalRepeatEnd; i++) {
              if (i != 0) {
                float angPhi = twist * i;
                float shift = rise * i;

                aRotateKernelROT.do_rotate_improved(size, partReal, angPhi, 0, 0);
                aRotateKernelROT.SetTextureShift(partReal);
                aRotateKernelROT.do_shift(size, partReal,
                                          make_float3(0, 0, shift));
                // sub.Add(partReal, partSum);
                hipLaunchKernelGGL(add, grid, block, 0, stream, size,
                                   (float *)partReal.GetDevicePtr(),
                                   (float *)partSum.GetDevicePtr());
              }
            }
          }

          if (aConfig.ApplySymmetry == Configuration::Symmetry_Rotational) {
            partSum.CopyDeviceToHost(sum);
            stringstream ss1;
            string outName = aConfig.Path + aConfig.Reference[ref] + "noSymm_";
            ss1 << outName << iter + 1 << ".em";
            emwrite(ss1.str(), sum, size, size, size);

            // partSum is now the averaged Particle without symmetry
            aRotateKernelROT.SetTexture(partSum);

            float angle = aConfig.RotationalAngleStep;

            for (int i = 1; i < aConfig.RotationalCount;
                 i++) { // i=0 is the average itself{ // ToDo Is this correct
                        // INDENT {}
              float angPhi = angle * i;

              aRotateKernelROT.do_rotate_improved(size, partReal, angPhi, 0, 0);
              // sub.Add(partReal, partSum);
              hipLaunchKernelGGL(add, grid, block, 0, stream, size,
                                 (float *)partReal.GetDevicePtr(),
                                 (float *)partSum.GetDevicePtr());
            }
          }

          if (aConfig.BFactor != 0) {
#if VERBOSE >= 1
            cout << "Apply B-factor of " << aConfig.BFactor << "..." << endl;
#endif
            partSum.CopyDeviceToHost(sum);
            stringstream ss1;
            string outName = aConfig.Path + aConfig.Reference[ref] + "noBfac_";
            ss1 << outName << iter + 1 << ".em";
            emwrite(ss1.str(), sum, size, size, size);

            // makecplx.MakeCplxWithSub(partSum, partCplx, 0);
            hipLaunchKernelGGL(makeCplxWithSub, grid, block, 0, stream, size,
                               (float *)partSum.GetDevicePtr(),
                               (float2 *)partCplx.GetDevicePtr(), 0);

            tempCplx.CopyDeviceToDevice(partCplx);
#if defined(__HIP_PLATFORM_HCC__)
            hipfftExecC2C(ffthandle, (hipfftComplex *)tempCplx.GetDevicePtr(),
                          (hipfftComplex *)tempCplx.GetDevicePtr(),
                          HIPFFT_FORWARD);
#else
            cufftExecC2C(ffthandle, (cufftComplex *)tempCplx.GetDevicePtr(),
                         (cufftComplex *)tempCplx.GetDevicePtr(),
                         CUFFT_FORWARD);
#endif

            // fft.FFTShift2(tempCplx, partCplx);
            hipLaunchKernelGGL(fftshift2, grid, block, 0, stream, size,
                               (float2 *)tempCplx.GetDevicePtr(),
                               (float2 *)partCplx.GetDevicePtr());

            float2 *particle = new float2[size * size * size];
            partCplx.CopyDeviceToHost(particle);

            for (int z = 0; z < size; z++) {
              for (int y = 0; y < size; y++) {
                for (int x = 0; x < size; x++) {
                  int dz = (z - size / 2);
                  int dy = (y - size / 2);
                  int dx = (x - size / 2);

                  float d = sqrt(dx * dx + dy * dy + dz * dz);

                  d = round(d);
                  d = d > (size / 2 - 1) ? (size / 2 - 1) : d;

                  float res = size / (d + 1) * aConfig.PixelSize;

                  float value = expf(-aConfig.BFactor / (4.0f * res * res));

                  size_t idx = z * size * size + y * size + x;
                  float2 pixel = particle[idx];
                  pixel.x *= value;
                  pixel.y *= value;
                  particle[idx] = pixel;
                }
              }
            }

            partCplx.CopyHostToDevice(particle);
            delete[] particle;
            // fft.FFTShift2(partCplx, tempCplx);
            hipLaunchKernelGGL(fftshift2, grid, block, 0, stream, size,
                               (float2 *)partCplx.GetDevicePtr(),
                               (float2 *)tempCplx.GetDevicePtr());

            partCplx.CopyDeviceToDevice(tempCplx);
#if defined(__HIP_PLATFORM_HCC__)
            hipfftExecC2C(ffthandle, (hipfftComplex *)partCplx.GetDevicePtr(),
                          (hipfftComplex *)partCplx.GetDevicePtr(),
                          HIPFFT_BACKWARD);
#else
            cufftExecC2C(ffthandle, (cufftComplex *)partCplx.GetDevicePtr(),
                         (cufftComplex *)partCplx.GetDevicePtr(),
                         CUFFT_INVERSE);
#endif

            // mul.Mul(1.0f / (float)size / (float)size / (float)size,
            // partCplx);
            hipLaunchKernelGGL(mul, grid, block, 0, stream, size,
                               1.0f / (float)size / (float)size / (float)size,
                               (float2 *)partCplx.GetDevicePtr());

            // makecplx.MakeReal(partCplx, partSum);
            hipLaunchKernelGGL(makeReal, grid, block, 0, stream, size,
                               (float2 *)partCplx.GetDevicePtr(),
                               (float *)partSum.GetDevicePtr());
          }

          partSum.CopyDeviceToHost(sum);
          stringstream ss1;
          string outName = aConfig.Path + aConfig.Reference[ref];
          ss1 << outName << iter + 1 << ".em";
          emwrite(ss1.str(), sum, size, size, size);

          partSumEven.CopyDeviceToHost(sum);
          stringstream ss2;
          ss2 << outName << iter + 1 << "Even.em";
          emwrite(ss2.str(), sum, size, size, size);
          partSumOdd.CopyDeviceToHost(sum);
          stringstream ss3;
          ss3 << outName << iter + 1 << "Odd.em";
          emwrite(ss3.str(), sum, size, size, size);

          partSumA.CopyDeviceToHost(sum);
          stringstream ss5;
          ss5 << outName << iter + 1 << "A.em";
          emwrite(ss5.str(), sum, size, size, size);
          partSumB.CopyDeviceToHost(sum);
          stringstream ss4;
          ss4 << outName << iter + 1 << "B.em";
          emwrite(ss4.str(), sum, size, size, size);
          delete[] sum;
        }
      }

      if (mpi_part == 0) {
        int *buffer = new int[aConfig.Reference.size()];
        for (int mpi = 1; mpi < mpi_size; mpi++) {
          MPI_Recv(buffer, aConfig.Reference.size(), MPI_INT, mpi, 0,
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          for (size_t i = 0; i < aConfig.Reference.size(); i++) {
            partsPerRef[i] += buffer[i];
          }
        }

        int totalUsed = 0;
        for (size_t i = 0; i < aConfig.Reference.size(); i++) {
          totalUsed += partsPerRef[i];
        }

        // Output statistics:
#if VERBOSE >= 1
        cout << "Total particles:   " << totalCount << endl;
        cout << "Ignored particles: " << totalCount - totalUsed << endl;
        cout << "Used particles:    " << totalUsed << endl;

        if (aConfig.MultiReference) {
          for (size_t i = 0; i < aConfig.Reference.size(); i++) {
            cout << "Used for ref" << i + 1 << ":     " << partsPerRef[i]
                 << endl;
          }
        }
#endif

        delete[] buffer;
      } else {
        MPI_Send(&partsPerRef[0], aConfig.Reference.size(), MPI_INT, 0, 0,
                 MPI_COMM_WORLD);
      }

#if defined(__HIP_PLATFORM_HCC__)
      hipfftDestroy(ffthandle);
#else
      cufftDestroy(ffthandle);
#endif
    }

    MPI_Barrier(MPI_COMM_WORLD);

    ////////////////////////////
    /// End of Add particles ///
    ////////////////////////////

    if (onlySumUp) {
      break;
    }
  }

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

