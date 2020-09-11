#ifndef MOTIVELIST_H
#define MOTIVELIST_H

#include "EMFile.h"
#include "../config/Config.h"

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

	string GetIndexCoding(Configuration::NamingConvention nc);
};

class MotiveList : public EMFile
{
public:
	MotiveList(string filename);

	motive GetAt(int index);

	void SetAt(int index, motive& m);
};
#endif //MOTIVELIST_H