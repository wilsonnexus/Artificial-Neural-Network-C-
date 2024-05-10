#ifndef MY_MATH_FUNCTIONS_H
#define MY_MATH_FUNCTIONS_H

#include <vector>
#include <limits>

using namespace std;

void printMatrix(const vector<vector<float>>& matrix);

vector<vector<float>> removeInitialBias(const vector<vector<float>>& a);

vector<float> addVectors(const vector<float>& v1, const vector<float>& v2);

vector<vector<float>> addMatrices(const vector<vector<float>>& m1, const vector<vector<float>>& m2);

vector<float> subtractVectors(const vector<float>& v1, const vector<float>& v2);

vector<vector<float>> subtractMatrices(const vector<vector<float>>& matrix1, const vector<vector<float>>& matrix2);

vector<float> multiplyVectorVector(vector<float>& vector1, vector<float>& vector2);

vector<vector<float>> multiplyVectorVectorTranspose(const vector<float>& vector1, const vector<float>& vector2);

vector<float> multiplyMatrixVector(vector<vector<float>>& matrix, vector<float>& Vector);

vector<vector<float>> multiplyVectorMatrix(const vector<float>& Vector, const vector<vector<float>>& matrix);

vector<vector<float>> multiplyMatrices(const vector<vector<float>>& matrix1, const vector<vector<float>>& matrix2);

vector<vector<float>> multiplyMatrixMatrixTranspose(const vector<vector<float>>& matrix1, const vector<vector<float>>& matrix2);

vector<vector<float>> transposeMatrix(const vector<vector<float>>& matrix);

vector<vector<float>> transposeVector(const vector<float>& vec);

vector<float> transposeVector(const vector<vector<float>>& matrix);

vector<vector<float>> elementWiseMult(float scalar, const vector<vector<float>>& matrix);

vector<float> elementWiseMult(const vector<float>& v1, const vector<float>& v2);

vector<vector<float>> elementWiseMult(const vector<vector<float>>& matrix, const vector<float>& vec);

vector<vector<float>> elementWiseMult(const vector<float>& vec, const vector<vector<float>>& matrix);

vector<vector<float>> elementWiseMult(const vector<vector<float>>& matrix1, const vector<vector<float>>& matrix2);

vector<float> scaleVector(vector<float> v, float scale);

vector<vector<float>> scaleMatrix(vector<vector<float>> mat, float scale);

vector<vector<float>> divideMatrixByScalar(const vector<vector<float>>& mat, float divisor);

#endif

