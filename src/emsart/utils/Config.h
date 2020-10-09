#ifndef CONFIG_H
#define CONFIG_H

#include "UtilsDefault.h"
#include <map>
#include <list>
#include "ConfigExceptions.h"
#include "../Kernels.h"

using namespace std;

//#define REFINE_MODE 1

namespace Configuration
{
#ifdef SUBVOLREC_MODE
	enum NamingConvention
	{
		NC_ParticleOnly, //Particle nr from line 4
		NC_TomogramParticle //Tomogram nr from line 5 and particle nr from line 6
	};
#endif


	//! Parse structured config files
	/*!
		Config files contains lines with name-value assignements in the form "<name> = <value>".
	   Trailing and leading whitespace is stripped. Parsed config entries are stored in
	   a symbol map.

	   Lines beginning with '#' are a comment and ignored.

	   Config files may be structured (to arbitrary depth). To start a new config sub group
	   (or sub section) use a line in the form of "<name> = (".
	   Subsequent entries are stured in the sub group, until a line containing ")" is found.

	   Values may reuse already defined names as a variable which gets expanded during
	   the parsing process. Names for expansion are searched from the current sub group
	   upwards. Finally the process environment is searched, so also environment
	   variables may be used as expansion symbols in the config file.
	*/
	//Parse structured config files
	class Config {
		public:
		enum FILE_SAVE_MODE
		{
			FSM_RAW,
			FSM_EM,
			FSM_MRC
		};
		enum CTF_MODE
		{
			CTFM_YES,
			CTFM_NO
		};
		enum FILE_READ_MODE
		{
			FRM_DM4,
			FRM_MRC
		};
		enum PROJ_NORMALIZATION_MODE
		{
			PNM_MEAN,
			PNM_STANDARD_DEV,
			PNM_NONE
		};
#ifdef REFINE_MODE
		enum GROUP_MODE
		{
			GM_BYGROUP,
			GM_MAXDIST,
			GM_MAXCOUNT
		};
#endif
	    private:
			//! Parse config file aConfigFile
			/*!
				If the process environment
				is provided, environment variables can be used as expansion symbols.
			*/
			//Parse config file 'aConfigFile'
			Config(string aConfigFile, int argc, char** argv, int mpiPart, char** appEnvp = 0);
			static Config* config;
			bool logAllowed;

		public:
			vector<int>	HipDeviceIDs;
			string	ProjectionFile;
			string	OutVolumeFile;
			string	MarkerFile;
			float	Lambda;
			float	AddTiltAngle;
			float	AddTiltXAngle;
			int		Iterations;
			dim3	RecDimensions;
			bool    UseFixPsiAngle;
			float   PsiAngle;
			float3  VolumeShift;
			int		ReferenceMarker;
			int		OverSampling;
			float	PhiAngle;
			float3  VoxelSize;
			float2  DimLength;
			float2  CutLength;
			float4  Crop;
			float4  CropDim;
			bool	SkipFilter;
			int     fourFilterLP;
			int     fourFilterLPS;
			int     fourFilterHP;
			int     fourFilterHPS;
			int     SIRTCount;
			//float   ProjNormVal;
			//bool    Filtered;
			CTF_MODE CtfMode;
			string  CtfFile;
			float	BadPixelValue;
			bool	CorrectBadPixels;
			float4	CTFBetaFac;
			bool	FP16Volume;
			bool	WriteVolumeAsFP16;
			float	ProjectionScaleFactor;
			PROJ_NORMALIZATION_MODE ProjectionNormalization;
			bool	WBP_NoSART;
			WbpFilterMethod WBPFilter;
			float	Cs;
			float	Voltage;
			float	MagAnisotropyAmount;
			float	MagAnisotropyAngleInDeg;
			bool	IgnoreZShiftForCTF;
			float	CTFSliceThickness;
			bool	DebugImages;
			bool	DownWeightTiltsForWBP;
			bool	SwitchCTFDirectionForIMOD;

#ifdef REFINE_MODE
			int SizeSubVol;
			float VoxelSizeSubVol;
			string MotiveList;
			float ScaleMotivelistShift;
			float ScaleMotivelistPosition;
			string Reference;
			int MaxShift;
			string ShiftOutputFile;
			GROUP_MODE GroupMode;
			int GroupSize;
			float MaxDistance;
			string CCMapFileName;
			float SpeedUpDistance;
#endif

#ifdef SUBVOLREC_MODE
			int SizeSubVol;
			float VoxelSizeSubVol;
			string MotiveList;
			float ScaleMotivelistShift;
			float ScaleMotivelistPosition;
			string SubVolPath;
			string ShiftInputFile;
			int BatchSize;
			int MaxShift;
			NamingConvention NamingConv;
#endif

            static Config& GetConfig();
            static Config& GetConfig(string aConfigFile, int argc, char** argv, int mpiPart, char** appEnvp = 0);

			~Config();

			FILE_SAVE_MODE GetFileSaveMode();

			FILE_READ_MODE GetFileReadMode();

			//! Get string config entry
			//Get string config entry
			string GetString(string aName);

			//! Get string config entry (for optional entries; does not throw exception if not found)
			//Get string config entry (for optional entries; does not throw exception if not found)
			string GetStringOptional(string aName);

			//! get boolean config entry
			/*!
				A value of Yes/yes/YES/true/True/TRUE leads to true,
				all other values leads to false.
			*/
			//get boolean config entry
			bool GetBool(string aName);

			//! get boolean config entry
			/*!
				A value of Yes/yes/YES/true/True/TRUE leads to true,
				all other values leads to false.
			*/
			//get boolean config entry
			bool GetBool(string aName, bool defaultVal);

			//! get double config entry; value is parsed using stringstream
			// get double config entry; value is parsed using stringstream
			double GetDouble(string name);

			//! get float config entry; value is parsed using stringstream
			// get float config entry; value is parsed using stringstream
			float GetFloat(string aName);

			//! get float config entry; value is parsed using stringstream
			// get float config entry; value is parsed using stringstream
			float GetFloat(string aName, float defaultVal);

			//! get int config entry; value is parsed using stringstream
			// get int config entry; value is parsed using stringstream
			int GetInt(string aName);

			//! get dim3 config entry; value is parsed using stringstream
			// get dim3 config entry; value is parsed using stringstream
			dim3 GetDim3(string aName);

			//! get float2 config entry; value is parsed using stringstream
			// get float2 config entry; value is parsed using stringstream
			float2 GetFloat2(string aName);

			//! get float2 config entry; value is parsed using stringstream
			// get float2 config entry; value is parsed using stringstream
			float2 GetFloat2(string aName, float2 defaultVal);

			//! get float3 config entry; value is parsed using stringstream
			// get float3 config entry; value is parsed using stringstream
			float3 GetFloat3(string aName);

			//! get float3 config entry; value is parsed using stringstream
			// get float3 config entry; value is parsed using stringstream
			float3 GetFloatOrFloat3(string aName);

			//! get float4 config entry; value is parsed using stringstream. If value not existent, default value is returned.
			// get float4 config entry; value is parsed using stringstream If value not existent, default value is returned.
			float4 GetFloat4(string aName, float4 defaultVal);

			//! get float3 config entry; value is parsed using stringstream
			// get float3 config entry; value is parsed using stringstream
			vector<int> GetVectorInt(string aName);

			//! get the symbol map (e.g. for iterating over all symbols)
			// get the symbol map (e.g. for iterating over all symbols)
			inline map<string, string>& GetSymbols() {
				return symbols;
			}

			string GetConfigFileName();

			//! get config sub group
			// get config sub group
			inline Config* GetGroup(string aName) {
				return groups[aName];
			}

			//! get config sub group map (e.g. for iterating over all groups)
			// get config sub group map (e.g. for iterating over all groups)
			inline map<string, Config*>& GetGroups() {
				return groups;
			}


		private:
			// private constructor for sub groups
			Config(string name, string parentDebugInfo);

			// helper functions for parsing
			void add(string name, string value);
			void split(string in, string& left, string& right, char c);
			void trim(string& s);
			void symbolExpand(string& s);
			void symbolExpand(map<string, string>& symbols, string& s);
			void envSymbolExpand(string& s);
			void replaceChar(string& str, char replace, char by);

			// config group symbol map
			map<string, string> symbols;

			// environment symbol map
			map<string, string> envSymbols;

			// config sub group map
			map<string, Config*> groups;

			// stack of config groups for parsing (only used in top config element)
			list<Config*> groupStack;

			// debug info used for logging messages
			string mDebugInfo;

			string mConfigFileName;
	};
} //end namespace Configuration

#endif //CONFIG_H
