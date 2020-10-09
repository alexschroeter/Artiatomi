#ifndef CTFFILE_H
#define CTFFILE_H

#include "EMFile.h"

class CtfFile : private EMFile
{
public:
	CtfFile(string aFileName);

	float GetMinDefocus(uint index);
	float GetMaxDefocus(uint index);
	float GetAstigmatismAngle(uint index);
};

#endif
