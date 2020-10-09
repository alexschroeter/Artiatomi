#ifndef WRITEBMP_H
#define WRITEBMP_H

#include "IODefault.h"



#define BITMAP_SIGNATURE 'MB'

typedef struct {
	unsigned int Size;
	unsigned int Reserved;
	unsigned int BitsOffset;
} BITMAP_FILEHEADER;

#define BITMAP_FILEHEADER_SIZE 14

typedef struct {
	unsigned int HeaderSize;
	int Width;
	int Height;
	unsigned short int Planes;
	unsigned short int BitCount;
	unsigned int Compression;
	unsigned int SizeImage;
	int PelsPerMeterX;
	int PelsPerMeterY;
	unsigned int ClrUsed;
	unsigned int ClrImportant;
} BITMAP_HEADER;


typedef struct {
	unsigned char Blue;
	unsigned char Green;
	unsigned char Red;
} Pixel;

//! Small and simple windows bitmap write function. Normalizes values to min/max. Float values
//Small and simple windows bitmap write function. Normalizes values to min/max. Float values
void writeBMP(std::string filename, float* aData, uint aWidth, uint aHeight);

//! Small and simple windows bitmap write function. Normalizes values to min/max. Double values
//Small and simple windows bitmap write function. Normalizes values to min/max. Double values
void writeBMP(std::string filename, double* aData, uint aWidth, uint aHeight);

//! Small and simple windows bitmap write function. Normalizes values to min/max. Int values
//Small and simple windows bitmap write function. Normalizes values to min/max. Int values
void writeBMP(std::string filename, int* aData, uint aWidth, uint aHeight);

//! Small and simple windows bitmap write function. Normalizes values to min/max. Ushort values
//Small and simple windows bitmap write function. Normalizes values to min/max. Ushort values
void writeBMP(std::string filename, ushort* aData, uint aWidth, uint aHeight);

//! Small and simple windows bitmap write function. Normalizes values to min/max. Char values
//Small and simple windows bitmap write function. Normalizes values to min/max. Char values
void writeBMP(std::string filename, char* aData, uint aWidth, uint aHeight);

#endif //WRITEBMP_H