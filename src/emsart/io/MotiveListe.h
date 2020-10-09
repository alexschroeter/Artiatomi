#ifndef MOTIVELIST_H
#define MOTIVELIST_H

#include "EMFile.h"
#ifdef SUBVOLREC_MODE
#include "../utils/Config.h"
#endif
#ifdef REFINE_MODE
#endif
#include "../utils/Config.h"

struct motive
{
	float ccCoeff;
	float xCoord;
	float yCoord;
	float partNr;
	float tomoNr;
	float partNrInTomo;
	float wedgeIdx;
	float x_Coord;
	float y_Coord;
	float z_Coord;
	float x_Shift;
	float y_Shift;
	float z_Shift;
	float x_ShiftBefore;
	float y_ShiftBefore;
	float z_ShiftBefore;
	float phi;
	float psi;
	float theta;
	float classNo;

	motive();
	
#ifdef SUBVOLREC_MODE
	string GetIndexCoding(Configuration::NamingConvention nc);
#endif

#ifdef REFINE_MODE
	bool isEqual(motive& m);
#endif
};

class MotiveList : public EMFile
{
	float binningFactorClick;
	float binningFactorShift;

	std::vector<int> groupIndices;

public:
	MotiveList(string filename, float aBinningFactorClick, float aBinningShift);

	motive GetAt(int index);

	void SetAt(int index, motive& m);

#ifdef REFINE_MODE
	float GetDistance(int aIndex1, int aIndex2);
	float GetDistance(motive& mot1, motive& mot2);
	std::vector<motive> GetNeighbours(int index, Configuration::Config& aConfig);
	std::vector<motive> GetNeighbours(int index, int count);
	std::vector<motive> GetNeighbours(int index, float maxDist);
	std::vector<motive> GetNeighbours(int groupNr);
	int GetGroupCount(Configuration::Config& aConfig);
	int GetGroupIdx(int groupNr);
	int GetGlobalIdx(motive& m);
#endif
};
#endif //MOTIVELIST_H