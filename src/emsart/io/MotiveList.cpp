#include "MotiveListe.h"
#include <sstream>
#include <algorithm>

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

MotiveList::MotiveList(string filename, float aBinningFactorClick, float aBinningShift)
	: EMFile(filename), binningFactorClick(aBinningFactorClick), binningFactorShift(aBinningShift)
{
	EMFile::OpenAndRead();
	EMFile::ReadHeaderInfo();

	for (int i = 0; i < DimY; i++)
	{
		motive m = GetAt(i);

		bool exists = false;
		for (int j = 0; j < groupIndices.size(); j++)
		{
			if (groupIndices[j] == m.classNo)
			{
				exists = true;
			}
		}

		if (!exists)
		{
			groupIndices.push_back(m.classNo);
		}
	}
}

motive MotiveList::GetAt(int index)
{
	motive m;
	memcpy(&m, _data + index * sizeof(m), sizeof(m));
	m.xCoord *= binningFactorClick;
	m.yCoord *= binningFactorClick;
	m.x_Coord *= binningFactorClick;
	m.y_Coord *= binningFactorClick;
	m.z_Coord *= binningFactorClick;
	m.x_Shift *= binningFactorShift;
	m.y_Shift *= binningFactorShift;
	m.z_Shift *= binningFactorShift;

	return m;
}

void MotiveList::SetAt(int index, motive& m)
{
	m.xCoord /= binningFactorClick;
	m.yCoord /= binningFactorClick;
	m.x_Coord /= binningFactorClick;
	m.y_Coord /= binningFactorClick;
	m.z_Coord /= binningFactorClick;
	m.x_Shift /= binningFactorShift;
	m.y_Shift /= binningFactorShift;
	m.z_Shift /= binningFactorShift;
	memcpy(_data + index * sizeof(m), &m, sizeof(m));
}

#ifdef REFINE_MODE
float MotiveList::GetDistance(int aIndex1, int aIndex2)
{
	motive mot1 = GetAt(aIndex1);
	motive mot2 = GetAt(aIndex2);

	return GetDistance(mot1, mot2);
}

float MotiveList::GetDistance(motive & mot1, motive & mot2)
{
	float x = mot1.x_Coord - mot2.x_Coord;
	float y = mot1.y_Coord - mot2.y_Coord;
	float z = mot1.z_Coord - mot2.z_Coord;

	float dist = sqrtf(x * x + y * y + z * z);

	return dist;
}

std::vector<motive> MotiveList::GetNeighbours(int index, Configuration::Config & aConfig)
{
	switch (aConfig.GroupMode)
	{
	case Configuration::Config::GM_BYGROUP:
		return GetNeighbours(index);
	case Configuration::Config::GM_MAXDIST:
		return GetNeighbours(index, aConfig.MaxDistance);
	case Configuration::Config::GM_MAXCOUNT:
		return GetNeighbours(index, aConfig.GroupSize);
	}
	return std::vector<motive>();
}

std::vector<motive> MotiveList::GetNeighbours(int index, int count)
{
	std::vector<motive> ret;
	std::vector<std::pair<float, int> > dists;

	for (int i = 0; i < DimY; i++) 
	{
		float dist = GetDistance(i, index);
		dists.push_back(pair<float, int>(dist, i));
	}

	//actual index is first element as it has distance zero!
	sort(dists.begin(), dists.end());

	for (int i = 0; i < count; i++)
	{
		ret.push_back(GetAt(dists[i].second));
	}

	return ret;
}

std::vector<motive> MotiveList::GetNeighbours(int index, float maxDist)
{
	std::vector<motive> ret;

	ret.push_back(GetAt(index)); //actual index is first element.
	for (int i = 0; i < DimY; i++)
	{
		if (GetDistance(i, index) <= maxDist && i != index)
		{
			ret.push_back(GetAt(i));
		}
	}

	return ret;
}
std::vector<motive> MotiveList::GetNeighbours(int groupNr)
{
	std::vector<motive> ret;

	int idx = GetGroupIdx(groupNr);
	if (idx < 0) return ret;

	for (int i = 0; i < DimY; i++)
	{
		motive m = GetAt(i);
		if (m.classNo == idx)
		{
			ret.push_back(m);
		}
	}

	return ret;
}

int MotiveList::GetGroupCount(Configuration::Config & aConfig)
{
	switch (aConfig.GroupMode)
	{
	case Configuration::Config::GM_BYGROUP:
		return (int)groupIndices.size();
	case Configuration::Config::GM_MAXDIST:
		return DimY;
	case Configuration::Config::GM_MAXCOUNT:
		return DimY;
	}
	return 0;
}

int MotiveList::GetGroupIdx(int groupNr)
{
	if (groupNr < 0 || groupNr >= groupIndices.size())
		return -1;

	return groupIndices[groupNr];
}
int MotiveList::GetGlobalIdx(motive & m)
{
	for (int i = 0; i < DimY; i++)
	{
		motive m2 = GetAt(i);
		if (m2.isEqual(m))
			return i;
	}
	return -1;
}
bool motive::isEqual(motive & m)
{
	bool eq = true;
	eq &= ccCoeff == m.ccCoeff;
	eq &= xCoord == m.xCoord;
	eq &= yCoord == m.yCoord;
	eq &= partNr == m.partNr;
	eq &= tomoNr == m.tomoNr;
	eq &= partNrInTomo == m.partNrInTomo;
	eq &= wedgeIdx == m.wedgeIdx;
	eq &= x_Coord == m.x_Coord;
	eq &= y_Coord == m.y_Coord;
	eq &= z_Coord == m.z_Coord;
	eq &= x_Shift == m.x_Shift;
	eq &= y_Shift == m.y_Shift;
	eq &= z_Shift == m.z_Shift;
	eq &= x_ShiftBefore == m.x_ShiftBefore;
	eq &= y_ShiftBefore == m.y_ShiftBefore;
	eq &= z_ShiftBefore == m.z_ShiftBefore;
	eq &= phi == m.phi;
	eq &= psi == m.psi;
	eq &= theta == m.theta;
	eq &= classNo == m.classNo;

	return eq;
}
#endif

#ifdef SUBVOLREC_MODE
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
#endif