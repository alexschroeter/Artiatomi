void computeRotMat(float phi, float psi, float theta, float rotMat[3][3]);
void multiplyRotMatrix(float m1[3][3], float m2[3][3], float out[3][3]);
void getEulerAngles(float matrix[3][3], float& phi, float& psi, float& theta);
bool checkIfClassIsToAverage(vector<int>& classes, int aClass);
