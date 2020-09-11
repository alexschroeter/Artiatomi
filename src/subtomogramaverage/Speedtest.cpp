/*******************************************************************************
 Subtomogram Averaging 

this code has originally been provided by Michael Kunz from the Frangakis group 
at the BMLS and has been adapted by Alexander Schröter for his Masterthesis.

The Goal of the Masterthesis is:
- Multi Platform GPU support (NVIDIA and AMD) 
	-> This is being achieved by the use of HIP (AMD)
- Performance Improvements
	-> Switch from C2C to R2C FFT
	-> Use of Callback functions
- Quality of live improvements
	-> -h
	-> commandline Options
		-> 

*******************************************************************************/


/* AS
Setting the compiler directive to DEBUG X will set the Debug level to:
0 = NO Debugging
1 = Basic Debugging
2 = Verbose Debugging
>10 = Everything you can think of
*/
#ifndef DEBUG
#define DEBUG 0
#endif

/* AS
Setting the compiler directive to TIME X will add Timings to specific sections 
relevant for performance measurement:
0 = No Timings
...
*/
#ifndef TIME
#define TIME 5
#endif

#define USE_MPI
#ifdef USE_MPI
#include <mpi.h>
#endif

#include <algorithm>
#include <time.h>
#include <iomanip>
#include <algorithm>
#include <map>
#include <omp.h>
#include <argp.h>

//#include "default.h"
#include "../config/Config.h"
#include "../io/MotiveListe.h"
#include "../io/EMFile.h"

#include "AvgProcess.h"
#include "../HelperFunctions.h"

#include "../hip/HipVariables.h"
#include "../hip/HipReducer.h"
#include "../hip/HipKernel.h"
#include "../hip/HipContext.h"

using namespace std;
using namespace Hip;

#define iterations 100

//#define round(x) (x >= 0 ? (int)(x + 0.5) : (int)(x - 0.5)) // TODO deprecated

void help(){
	cout << endl;
	//cout << "Usage: " << argv[0] << endl;
	cout << "    The following optional options override the configuration file:" << endl;
	cout << "    Options: " << endl;
	cout << "    -u FILENAME:   Use a user defined configuration file." << endl;
	cout << "    -h:            Show this text." << endl;

	cout << ("\nPress <Enter> to exit...");

	char c = cin.get();
	exit(-1);
}

int main(int argc, char* argv[]){
/* 
 * The majority of the main loop is setting up the work environment (reading 
 * data, getting the configuration)
 *
 * The 'Work' happens in the AvgProcess
 *
 * ToDo it would be nice to get more structure
 */

	/* 
	 * Setting up the Environment
	 * Preparing MPI and the Configuration
	 */

	/**************************************************************************/
	/**************************************************************************/
	/*																		  */
	/*																		  */
	/*						Message Parsing Interface						  */
	/*																		  */
	/*																		  */
	/**************************************************************************/
	/**************************************************************************/

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

	for (int i = 1; i < argc; i++){
		string temp(argv[i]);
		if (temp == "-sumup"){
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
	#if DEBUG >= 1
	printf("MPI process %d of %d on PC %s\n", mpi_part, mpi_size, mpi_name);
	#endif

	if (mpi_part == 0){
		hostnames.push_back(string(mpi_name));
		for (int i = 1; i < mpi_size; i++){
			char tempname[mpi_max_name_size];
			MPI_Recv(tempname, mpi_max_name_size, MPI_CHAR, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			hostnames.push_back(string(tempname));
		}

		// printf("Found %d hostnames\n", hostnames.size());

		for (int i = 0; i < mpi_size; i++){
			bool exists = false;
			for (int h = 0; h < singlehostnames.size(); h++){
				if (hostnames[i] == singlehostnames[h]){
					exists = true;
				}
			}
			if (!exists){
				singlehostnames.push_back(hostnames[i]);
			}
		}

		// sort host names alphabetically to obtain deterministic host IDs
		sort(singlehostnames.begin(), singlehostnames.end());

		for (int i = 1; i < mpi_size; i++){
			int host_id;
			int host_rank = 0;
			int offset = 0;

			string hostname = hostnames[i];

			for (int h = 0; h < singlehostnames.size(); h++){
				if (singlehostnames[h] == hostname){
					host_id = h;
					break;
				}
			}

			for (int h = 0; h < i; h++){
				if (hostnames[h] == hostname){
					host_rank++;
				}
			}

			for (int h = 0; h < host_id; h++){
				for (int n = 0; n < hostnames.size(); n++){
					if (hostnames[n] == singlehostnames[h]){
						offset++;
					}
				}
			}

			MPI_Send(&host_id, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
			MPI_Send(&host_rank, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
			MPI_Send(&offset, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
		}

		for (int h = 0; h < singlehostnames.size(); h++){
			if (singlehostnames[h] == string(mpi_name)){
				mpi_host_id = h;
				break;
			}
		}

		for (int h = 0; h < mpi_host_id; h++){
			for (int n = 0; n < hostnames.size(); n++){
				if (hostnames[n] == singlehostnames[h]){
					mpi_offset++;
				}
			}
		}
		mpi_host_rank = 0;

	} else {
		MPI_Send(mpi_name, mpi_max_name_size, MPI_CHAR, 0, 0, MPI_COMM_WORLD);

		MPI_Recv(&mpi_host_id, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(&mpi_host_rank, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(&mpi_offset, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}

	#if DEBUG >= 1
	printf("Host ID: %d; host rank: %d; offset: %d; global rank: %d; name: %s\n", mpi_host_id, mpi_host_rank, mpi_offset, mpi_part, mpi_name);
	fflush(stdout);
	#endif

	MPI_Barrier(MPI_COMM_WORLD);
	#endif

	/* AS *********************************************************************/
	/**************************************************************************/
	/*																		  */
	/*																		  */
	/*						User Interface									  */
	/*																		  */
	/*																		  */
	/**************************************************************************/
	/**************************************************************************/

	// bool C2C = false;

	// int opt;
	// while ((opt = getopt (argc, argv, "huc")) != -1)
	// {
	// 	switch (opt)
	// 	{
	// 		case 'h':
	// 			if (mpi_part==0) {
	// 				help();
	// 			}
	// 			MPI_Finalize(); // finalizing MPI interface
	// 			return 0;
	// 		case 'u':
	// 			break;
	// 		case 'c':
	// 				AS
	// 				The original code used a Complex-to-Complex Fast Fourier Transform (FFT). 
	// 				During the performance optimization this was changed to a Real-to-Complex FFT
	// 				You can set the commandline option "-c" to use the old Version
	// 				ToDo: -c should be changed to -C2C
				
	// 			C2C=true;
	// 			break;
	// 		default:
	// 			if (mpi_part == 0){
	// 				help();
	// 			}
	// 			MPI_Finalize(); // finalizing MPI interface
	// 			return 0;
	// 	}
	// }

	// // if parameters are not given via parameters input is required
	// if (mpi_part==0){
	// 	printf("Unknown Option. The first integral has been selected.\n");
	// }

	//////////////////////////////////////////////////////////////////////

	int c;

	/* Flag set by ‘--verbose’. */
	static int verbose_flag = 0;
	static int c2c_flag = 1;
	string config_path = "average.cfg";
  	
  	while (true)
    {
		static struct option long_options[] =
		{
			/* These options set a flag. */
			{"verbose",	no_argument,       	&verbose_flag, 	1},
			{"brief",	no_argument,       	&verbose_flag, 	0},
			/* These options don’t set a flag.
			We distinguish them by their indices. */
			{"config",	required_argument,	nullptr, 		'u'},
			{"C2C",     no_argument,       	&c2c_flag, 		'1'},
			{"R2C",     no_argument,       	&c2c_flag, 		'0'},
			{"time",  	required_argument,  nullptr, 		't'},
			{nullptr, 	no_argument, 		nullptr, 		0}
		};
		/* getopt_long stores the option index here. */
		int option_index = 0;

		c = getopt_long (argc, argv, "u:t:",
		               long_options, &option_index);

		/* Detect the end of the options. */
		if (c == -1)
			break;

		switch (c)
		{
			case 'u':
				config_path = optarg;
				printf("Configuration file is located at %s.\n", optarg);
				break;

			case 't':
				printf ("The time measurement level has been set to `%i'\n", atoi(optarg));
				break;

			case '?':
				/* getopt_long already printed an error message. */
				break;

			default:
				help();
		}
    }


	/**************************************************************************/
	/**************************************************************************/
	/*																		  */
	/*																		  */
	/*						Configuration									  */
	/*																		  */
	/*																		  */
	/**************************************************************************/
	/**************************************************************************/

	Configuration::Config aConfig = Configuration::Config::GetConfig(config_path, argc, argv, mpi_part, NULL);
	Hip::HipContext* ctx = Hip::HipContext::CreateInstance(aConfig.CudaDeviceIDs[mpi_part], hipDeviceScheduleSpin);


	clock_t t_start, t_end; 

	size_t sizes;
	for (int size = 7; size < 8; size++){

		sizes = 32*size;//pow(2, size);
		//cout << sizes << endl;

		for (int iter = aConfig.StartIteration; iter < aConfig.EndIteration; iter++){
		/*
		 * beginning of the main loop
		 * the config file defines how many time the routine should be performed
		 */

		    srand((unsigned)time(NULL));

			float* mask_data = (float*)malloc(sizeof(float)*sizes*sizes*sizes);
			for (int i = 0; i < sizes*sizes*sizes; i++){
				mask_data[i] = (float)rand()/RAND_MAX;
			}

			float* maskcc_data = (float*)malloc(sizeof(float)*sizes*sizes*sizes);
			for (int i = 0; i < sizes*sizes*sizes; i++){
				maskcc_data[i] = (float)rand()/RAND_MAX;
			}

			float* multiref_data = (float*)malloc(sizeof(float)*sizes*sizes*sizes);
			for (int i = 0; i < sizes*sizes*sizes; i++){
				multiref_data[i] = (float)rand()/RAND_MAX;
			}

			float* filter_data = (float*)malloc(sizeof(float)*sizes*sizes*sizes);
			for (int i = 0; i < sizes*sizes*sizes; i++){
				filter_data[i] = (float)rand()/RAND_MAX;
			}

			float* wedge_data = (float*)malloc(sizeof(float)*sizes*sizes*sizes);
			for (int i = 0; i < sizes*sizes*sizes; i++){
				wedge_data[i] = (float)rand()/RAND_MAX;
			}

			float* part_data = (float*)malloc(sizeof(float)*sizes*sizes*sizes);
			for (int i = 0; i < sizes*sizes*sizes; i++){
				part_data[i] = (float)rand()/RAND_MAX;
			}

			stringstream ssml;
			ssml << aConfig.Path << aConfig.MotiveList << iter << ".em";
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
			ssref << aConfig.Path << aConfig.Reference[0] << iter << ".em";
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

			int size = ref.DimX;
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


			p = new AvgProcessC2C(sizes, 0, ctx, (float*)mask_data, (float*)multiref_data, (float*)maskcc_data, aConfig.PhiAngIter, aConfig.PhiAngIncr, aConfig.AngIter, aConfig.AngIncr, aConfig.BinarizeMask, aConfig.RotateMaskCC, aConfig.UseFilterVolume, aConfig.LinearInterpolation);

			clock_t t_total = 0;
		
			for (int iteration = 0; iteration < iterations; iteration++){

				//int oldIndex = maxVals_t::getIndex(size, (int)mot.x_Shift, (int)mot.y_Shift, (int)mot.z_Shift);		
				int oldIndex = 0;

				t_start = clock();

				maxVals_t v = p->execute((float*)part_data, (float*)wedge_data, (float*)filter_data, mot.phi, mot.psi, mot.theta, (float)aConfig.HighPass, (float)aConfig.LowPass, (float)aConfig.Sigma, make_float3(mot.x_Shift, mot.y_Shift, mot.z_Shift), aConfig.CouplePhiToPsi, aConfig.ComputeCCValOnly, oldIndex);

				t_end = clock();

				t_total += t_end - t_start;
			}

			ctx->Synchronize();
			delete(p);
			printf("Iterations: %i\tSize: %i\tSeconds: %f\tVersion: C2C\n",iterations , sizes, (double) (t_total) / CLOCKS_PER_SEC);
					
			hipDeviceReset();


			//p = new AvgProcessR2C(sizes, 0, ctx, (float*)mask_data, (float*)multiref_data, (float*)maskcc_data, aConfig.PhiAngIter, aConfig.PhiAngIncr, aConfig.AngIter, aConfig.AngIncr, aConfig.BinarizeMask, aConfig.RotateMaskCC, aConfig.UseFilterVolume, aConfig.LinearInterpolation);

			t_total = 0;
			//clock_t t_total = 0;

			for (int iteration = 0; iteration < iterations; iteration++){

				//int oldIndex = maxVals_t::getIndex(size, (int)mot.x_Shift, (int)mot.y_Shift, (int)mot.z_Shift);		
				int oldIndex = 0;

				t_start = clock();

				maxVals_t v = p->execute((float*)part_data, (float*)wedge_data, (float*)filter_data, mot.phi, mot.psi, mot.theta, (float)aConfig.HighPass, (float)aConfig.LowPass, (float)aConfig.Sigma, make_float3(mot.x_Shift, mot.y_Shift, mot.z_Shift), aConfig.CouplePhiToPsi, aConfig.ComputeCCValOnly, oldIndex);

				t_end = clock();

				t_total += t_end - t_start;
			}
			
			ctx->Synchronize();
			delete(p);
			printf("Iterations: %i\tSize: %i\tSeconds: %f\tVersion: R2C\n",iterations , sizes, (double) (t_total) / CLOCKS_PER_SEC);
			
			hipDeviceReset();

		}
	}
	//cudaProfilerStop();

	MPI_Finalize();
	return 0;
}



void computeRotMat(float phi, float psi, float theta, float rotMat[3][3]){
	/* TODO
	 *
	 * What is all this stuff?
	 */
	int i, j;
	float sinphi, sinpsi, sintheta;	/* sin of rotation angles */
	float cosphi, cospsi, costheta;	/* cos of rotation angles */

	float angles[] = { 0, 30, 45, 60, 90, 120, 135, 150, 180, 210, 225, 240, 270, 300, 315, 330 };
	float angle_cos[16];
	float angle_sin[16];

	/* ToDo Values dont need to be calculated every time the computeRotMat function is call */
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

	for (i = 0, j = 0; i<16; i++){
		if (angles[i] == phi){
			cosphi = angle_cos[i];
			sinphi = angle_sin[i];
			j = 1;
		}
	}

	if (j < 1){
		phi = phi * (float)M_PI / 180.0f;
		cosphi = cos(phi);
		sinphi = sin(phi);
	}

	for (i = 0, j = 0; i<16; i++){
		if (angles[i] == psi){
			cospsi = angle_cos[i];
			sinpsi = angle_sin[i];
			j = 1;
		}
	}

	if (j < 1){
		psi = psi * (float)M_PI / 180.0f;
		cospsi = cos(psi);
		sinpsi = sin(psi);
	}

	for (i = 0, j = 0; i<16; i++){
		if (angles[i] == theta){
			costheta = angle_cos[i];
			sintheta = angle_sin[i];
			j = 1;
		}
	}

	if (j < 1){
		theta = theta * (float)M_PI / 180.0f;
		costheta = cos(theta);
		sintheta = sin(theta);
	}

	/* ToDo This can be partial results can be reused */
	/* calculation of rotation matrix */
	rotMat[0][0] = cospsi*cosphi - costheta*sinpsi*sinphi;
	rotMat[1][0] = sinpsi*cosphi + costheta*cospsi*sinphi;
	rotMat[2][0] = sintheta*sinphi;
	rotMat[0][1] = -cospsi*sinphi - costheta*sinpsi*cosphi;
	rotMat[1][1] = -sinpsi*sinphi + costheta*cospsi*cosphi;
	rotMat[2][1] = sintheta*cosphi;
	rotMat[0][2] = sintheta*sinpsi;
	rotMat[1][2] = -sintheta*cospsi;
	rotMat[2][2] = costheta;
}

void multiplyRotMatrix(float m1[3][3], float m2[3][3], float out[3][3]){
	/* ToDo Replace with Laerman Multiplication Algorithm 
	 *
	 * https://stackoverflow.com/questions/10827209/ladermans-3x3-matrix-multiplication-with-only-23-multiplications-is-it-worth-i 
	 */
	out[0][0] = (float)((double)m1[0][0] * (double)m2[0][0] + (double)m1[1][0] * (double)m2[0][1] + (double)m1[2][0] * (double)m2[0][2]);
	out[1][0] = (float)((double)m1[0][0] * (double)m2[1][0] + (double)m1[1][0] * (double)m2[1][1] + (double)m1[2][0] * (double)m2[1][2]);
	out[2][0] = (float)((double)m1[0][0] * (double)m2[2][0] + (double)m1[1][0] * (double)m2[2][1] + (double)m1[2][0] * (double)m2[2][2]);
	out[0][1] = (float)((double)m1[0][1] * (double)m2[0][0] + (double)m1[1][1] * (double)m2[0][1] + (double)m1[2][1] * (double)m2[0][2]);
	out[1][1] = (float)((double)m1[0][1] * (double)m2[1][0] + (double)m1[1][1] * (double)m2[1][1] + (double)m1[2][1] * (double)m2[1][2]);
	out[2][1] = (float)((double)m1[0][1] * (double)m2[2][0] + (double)m1[1][1] * (double)m2[2][1] + (double)m1[2][1] * (double)m2[2][2]);
	out[0][2] = (float)((double)m1[0][2] * (double)m2[0][0] + (double)m1[1][2] * (double)m2[0][1] + (double)m1[2][2] * (double)m2[0][2]);
	out[1][2] = (float)((double)m1[0][2] * (double)m2[1][0] + (double)m1[1][2] * (double)m2[1][1] + (double)m1[2][2] * (double)m2[1][2]);
	out[2][2] = (float)((double)m1[0][2] * (double)m2[2][0] + (double)m1[1][2] * (double)m2[2][1] + (double)m1[2][2] * (double)m2[2][2]);
}

void getEulerAngles(float matrix[3][3], float& phi, float& psi, float& theta){
	/* 
	 *
	 * 
	 */
	theta = acos(matrix[2][2])*180.0f / (float)M_PI;

	if (matrix[2][2] > 0.999){
		float sign = matrix[1][0] > 0 ? 1.0f : -1.0f;
		//matrix[2][2] < 0 ? -sign : sign;
		phi = sign * acos(matrix[0][0])*180.0f / (float)M_PI;
		psi = 0.0f;
	} else {
		phi = atan2(matrix[2][0], matrix[2][1]) * 180.0f / (float)M_PI;
		psi = atan2(matrix[0][2], -matrix[1][2]) * 180.0f / (float)M_PI;
		//phi = atan2(matrix[0][2], matrix[2][1]) * 180.0f / (float)M_PI;
		//psi = atan2(matrix[2][0], -matrix[1][2]) * 180.0f / (float)M_PI;
	}
}

bool checkIfClassIsToAverage(vector<int>& classes, int aClass){
	/* 
	 *
	 * 
	 */
	if (classes.size() == 0){
		return true;
	}

	for (int c : classes){
		if (c == aClass){
			return true;
		}
	}
	return false;
}
