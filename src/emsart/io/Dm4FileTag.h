#ifndef DM4FILETAG_H
#define DM4FILETAG_H

#include "IODefault.h"

enum DataTypesTag
{
	DTTI2 = 2,
	DTTI4 = 3,
	DTTUI2 = 4,
	DTTUI4 = 5,
	DTTF4 = 6,
	DTTF8 = 7,
	DTTI1 = 8,
	DTTCHAR = 9,
	DTTI1_2 = 10,
	DTTUI8 = 11,
};

enum DataTypesImage
{
	DTT_UNKNOWN   = 0,
	DTT_I2        = 1,
	DTT_F4        = 2,
	DTT_C8        = 3,
	DTT_OBSOLETE  = 4,
	DTT_C4        = 5,
	DTT_UI1       = 6,
	DTT_I4        = 7,
	DTT_RGB_4UI1  = 8,
	DTT_I1        = 9,
	DTT_UI2       = 10,
	DTT_UI4       = 11,
	DTT_F8        = 12,
	DTT_C16       = 13,
	DTT_BINARY    = 14,
	DTT_RGBA_4UI1 = 23
};

enum TagTypes
{
	SingleEntryTag,
	StructTag,
	ArrayTag,
	ArrayStructTag
};

//!  Dm4FileTag represents a gatan *.dm4 tag.
/*!
	\author Michael Kunz
	\date   September 2011
	\version 1.0
*/
class Dm4FileTag
{
public:
	Dm4FileTag();
	virtual ~Dm4FileTag();

	std::string Name;
	uint SizeInfoArray;
	uint* InfoArray;
	char* Data;
	uint SizeData;
	TagTypes tagType;

	uint GetSize(uint typeId);
	uint GetSizePixel(uint typeId);
	//void PrintValues(std::ostream& aStream);

	friend std::ostream& operator<<(std::ostream& stream, Dm4FileTag& tag);

	void PrintSingleValue(std::ostream& aStream, uint aType, uint aStartPos = 0);
	int GetSingleValueInt(uint aType, uint aStartPos = 0);
	float GetSingleValueFloat(uint aType, uint aStartPos = 0);
	double GetSingleValueDouble(uint aType, uint aStartPos = 0);
	std::string GetSingleValueString(uint aType, uint aStartPos = 0);

	int GetStructValueAsInt(uint aIndex);
	void FreeData();
	/*uint GetValueAsUint();
	float GetValueAsFloat();
	double GetValueAsDouble();*/


};


#endif
