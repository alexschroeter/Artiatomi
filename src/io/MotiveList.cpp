#include "MotiveListe.h"
#include <sstream>

motive::motive() :
	ccCoeff(0),
	xCoord(0),
	yCoord(0),
	partNr(0),
	tomoNr(0),
	partNrInTomo(0),
	wedgeIdx(0),
	x_Coord(0),
	y_Coord(0),
	z_Coord(0),
	x_Shift(0),
	y_Shift(0),
	z_Shift(0),
	x_ShiftBefore(0),
	y_ShiftBefore(0),
	z_ShiftBefore(0),
	phi(0),
	psi(0),
	theta(0),
	classNo(0)
{

}

MotiveList::MotiveList(string filename)
	: EMFile(filename)
{
	EMFile::OpenAndRead();
	EMFile::ReadHeaderInfo();
}

motive MotiveList::GetAt(int index)
{
	motive m;
	memcpy(&m, _data + index * sizeof(m), sizeof(m));
	return m;
}

void MotiveList::SetAt(int index, motive& m)
{
	memcpy(_data + index * sizeof(m), &m, sizeof(m));
}

string motive::GetIndexCoding(Configuration::NamingConvention nc)
{
	stringstream ss;
	switch (nc)
	{
	case Configuration::NC_ParticleOnly:
		ss << partNr;
		break;
	case Configuration::NC_TomogramParticle:
		ss << tomoNr << "_" << partNrInTomo;
		break;
	default:
		break;
	}
	return ss.str();
}