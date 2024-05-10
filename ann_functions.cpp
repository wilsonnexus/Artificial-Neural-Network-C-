#include "ann_functions.h"
#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

void printMatrix(const vector<vector<float>>& matrix) {
    for (const auto& row : matrix) {
        for (const auto& val : row) {
            cout << val << " ";
        }
        cout << endl;
    }
}

vector<vector<float>> removeInitialBias(const vector<vector<float>>& a) {
    vector<vector<float>> newA;
    int size = a.size();

    for (int i = 0; i < size - 1; ++i) {
        const auto& innerVec = a[i];
        vector<float> newInnerVec(innerVec.begin() + 1, innerVec.end());
        newA.push_back(newInnerVec);
    }

    // Keep the last vector as-is
    newA.push_back(a.back());

    return newA;
}

vector<float> addVectors(const vector<float>& v1, const vector<float>& v2) {
    // Ensure both vectors are of the same size
    if (v1.size() != v2.size()) {
        throw "Vectors must be of the same size to add them together";
    }

    // Create a result vector with the same size as the input vectors
    vector<float> result(v1.size());

    // Add corresponding elements from both vectors and store the result in the result vector
    for (size_t i = 0; i < v1.size(); ++i) {
        result[i] = v1[i] + v2[i];
    }

    return result;
}

// Subtract corresponding elements of two vectors
vector<float> subtractVectors(const vector<float>& v1, const vector<float>& v2) {
    if (v1.size() != v2.size()) {
        cerr << "Error: Vectors must be of the same size for subtraction." << endl;
        return {};
    }

    vector<float> result(v1.size());
    for (size_t i = 0; i < v1.size(); ++i) {
        result[i] = v1[i] - v2[i];
    }
    return result;
}

vector<vector<float>> subtractMatrices(const vector<vector<float>>& matrix1, const vector<vector<float>>& matrix2) {
    // Check if dimensions are compatible
    if (matrix1.size() != matrix2.size() || matrix1[0].size() != matrix2[0].size()) {
        cerr << "Error: Matrices must have the same dimensions for subtraction." << endl;
        return {};
    }

    vector<vector<float>> result(matrix1.size(), vector<float>(matrix1[0].size()));
    for (size_t i = 0; i < matrix1.size(); ++i) {
        for (size_t j = 0; j < matrix1[i].size(); ++j) {
            result[i][j] = matrix1[i][j] - matrix2[i][j];
        }
    }
    return result;
}

// Remove the first element from a vector
vector<float> removeFirstElement(const vector<float>& vec) {
    if (vec.empty()) {
        cerr << "Error: Cannot remove first element from an empty vector." << endl;
        return {};
    }

    vector<float> result(vec.begin() + 1, vec.end());
    return result;
}


vector<vector<float>> addMatrices(const vector<vector<float>>& m1, const vector<vector<float>>& m2) {
    int rows1 = m1.size();
    int cols1 = m1[0].size();
    int rows2 = m2.size();
    int cols2 = m2[0].size();

    int maxRows = max(rows1, rows2);
    int maxCols = max(cols1, cols2);

    // Create a result matrix with the maximum dimensions
    vector<vector<float>> result(maxRows, vector<float>(maxCols, 0.0));

    // Add corresponding elements from both matrices and store the result in the result matrix
    for (int i = 0; i < maxRows; ++i) {
        for (int j = 0; j < maxCols; ++j) {
            float val1 = (i < rows1&& j < cols1) ? m1[i][j] : 0.0;
            float val2 = (i < rows2&& j < cols2) ? m2[i][j] : 0.0;
            result[i][j] = val1 + val2;
        }
    }

    return result;
}

vector<float> multiplyVectorVector(vector<float>& vector1, vector<float>& vector2) {
    // Check if the sizes of both vectors are the same
    if (vector1.size() != vector2.size()) {
        cerr << "Invalid vector-vector multiplication: Incompatible vector sizes." << endl;
        return {}; // Return an empty result vector
    }

    vector<float> result(vector1.size(), 0);

    // Perform element-wise multiplication
    for (int i = 0; i < vector1.size(); ++i) {
        result[i] = vector1[i] * vector2[i];
    }

    return result;
}

vector<vector<float>> multiplyVectorVectorTranspose(const vector<float>& vector1, const vector<float>& vector2) {
    int size1 = vector1.size();
    int size2 = vector2.size();

    // Check if one of the vectors has only one value
    if (size1 == 1 || size2 == 1) {
        // In this case, perform regular element-wise multiplication
        vector<vector<float>> result(size1, vector<float>(size2, 0.0));
        for (int i = 0; i < size1; ++i) {
            for (int j = 0; j < size2; ++j) {
                result[i][j] = vector1[i] * vector2[j];
            }
        }
        return result;
    }

    // Otherwise, perform transpose multiplication
    vector<vector<float>> result(size2, vector<float>(size1, 0.0));
    for (int i = 0; i < size2; ++i) {
        for (int j = 0; j < size1; ++j) {
            result[i][j] = vector1[j] * vector2[i];
        }
    }
    return result;
}



// Function to multiply a 2D vector (matrix) with a 1D vector
vector<float> multiplyMatrixVector(vector<vector<float>>& matrix, vector<float>& Vector) {
    //cout << matrix[0][0] << endl;
    //cout << matrix[0].size() << endl;
    vector<float> result(matrix.size(), 0);
    //cout << Vector.size() << endl;

    // Check if the number of columns in the matrix is equal to the size of the vector
    if (matrix[0].size() != Vector.size()) {
        cerr << "Invalid matrix-vector multiplication: Incompatible matrix and vector sizes." << endl;
        return result; // Return an empty result vector
    }

    // Perform matrix-vector multiplication
    for (int i = 0; i < matrix.size(); ++i) {
        for (int j = 0; j < matrix[i].size(); ++j) {

            result[i] += matrix[i][j] * Vector[j];
        }
    }

    return result;
}

vector<vector<float>> multiplyVectorMatrix(const vector<float>& Vector, const vector<vector<float>>& matrix) {
    int numRows = matrix.size();
    int numCols = matrix[0].size();
    int vectorSize = Vector.size();

    // Check if the size of the vector is equal to the number of rows in the matrix
    if (vectorSize != numRows) {
        cerr << "Invalid vector-matrix multiplication: Incompatible vector and matrix sizes." << endl;
        return {};
    }

    vector<vector<float>> result(1, vector<float>(numCols, 0)); // Result is a 1 x numCols matrix

    // Perform vector-matrix multiplication
    for (int j = 0; j < numCols; ++j) {
        for (int i = 0; i < numRows; ++i) {
            result[0][j] += Vector[i] * matrix[i][j];
        }
    }

    return result;
}

vector<vector<float>> multiplyMatrices(const vector<vector<float>>& matrix1, const vector<vector<float>>& matrix2) {
    // Check if either matrix is empty
    if (matrix1.empty() || matrix2.empty()) {
        cerr << "Invalid matrix multiplication: One or both matrices are empty." << endl;
        return {}; // Return an empty result matrix
    }

    // If matrix2 is 1x1, treat it as a scalar
    if (matrix2.size() == 1 && matrix2[0].size() == 1) {
        float scalar = matrix2[0][0];
        vector<vector<float>> result(matrix1.size(), vector<float>(matrix1[0].size(), 0));

        // Perform scalar multiplication
        for (size_t i = 0; i < matrix1.size(); ++i) {
            for (size_t j = 0; j < matrix1[0].size(); ++j) {
                result[i][j] = matrix1[i][j] * scalar;
            }
        }

        return result;
    }

    // Check if the number of columns in the first matrix is equal to the number of rows in the second matrix
    if (matrix1[0].size() != matrix2.size()) {
        cerr << "Invalid matrix multiplication: Incompatible matrix sizes." << endl;
        return {}; // Return an empty result matrix
    }

    // Create a vector to store the result of the multiplication
    vector<vector<float>> result(matrix1.size(), vector<float>(matrix2[0].size(), 0));

    // Perform matrix multiplication
    for (size_t i = 0; i < matrix1.size(); ++i) {
        for (size_t j = 0; j < matrix2[0].size(); ++j) {
            for (size_t k = 0; k < matrix2.size(); ++k) {
                result[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }

    return result;
}


vector<vector<float>> multiplyMatrixMatrixTranspose(const vector<vector<float>>& matrix1, const vector<vector<float>>& matrix2) {
    int rows1 = matrix1.size();
    int cols1 = matrix1[0].size();
    int rows2 = matrix2.size();
    int cols2 = matrix2[0].size();

    // Check if the dimensions are compatible for matrix multiplication
    if (cols1 != cols2) {
        throw "Incompatible matrix dimensions for multiplication";
    }

    // Initialize the result matrix
    vector<vector<float>> result(rows1, vector<float>(rows2, 0.0));

    // Perform matrix multiplication
    for (int i = 0; i < rows1; ++i) {
        for (int j = 0; j < rows2; ++j) {
            for (int k = 0; k < cols1; ++k) {
                result[i][j] += matrix1[i][k] * matrix2[j][k];
            }
        }
    }

    return result;
}



vector<vector<float>> transposeMatrix(const vector<vector<float>>& matrix) {
    int rows = matrix.size();
    int cols = matrix[0].size();

    // Create a new matrix to store the transpose
    vector<vector<float>> transpose(cols, vector<float>(rows));

    // Copy elements from the original matrix to the transpose matrix
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            transpose[j][i] = matrix[i][j];
        }
    }

    return transpose;
}

vector<vector<float>> transposeVector(const vector<float>& vec) {
    int size = vec.size();

    // Create a new matrix to store the transpose
    vector<vector<float>> transpose(size, vector<float>(1));

    // Copy elements from the vector to the transpose matrix
    for (int i = 0; i < size; ++i) {
        transpose[i][0] = vec[i];
    }

    return transpose;
}


vector<float> transposeVector(const vector<vector<float>>& matrix) {
    int size = matrix.size(); // Number of columns

    // Create a new vector to store the transpose
    vector<float> transpose(size);

    // Copy elements from the matrix to the transpose vector
    for (int i = 0; i < size; ++i) {
        transpose[i] = matrix[i][0];
    }

    return transpose;
}

vector<vector<float>> elementWiseMult(float scalar, const vector<vector<float>>& matrix) {
    int numRows = matrix.size();
    int numCols = matrix[0].size();

    vector<vector<float>> result(numRows, vector<float>(numCols));

    // Perform element-wise multiplication
    for (int i = 0; i < numRows; ++i) {
        for (int j = 0; j < numCols; ++j) {
            result[i][j] = scalar * matrix[i][j];
        }
    }

    return result;
}


// Element-wise multiplication of two vectors
vector<float> elementWiseMult(const vector<float>& v1, const vector<float>& v2) {

    // Check that vectors are same size
    if (v1.size() != v2.size()) {
        throw "Vectors must be same size for element-wise multiplication";
    }

    vector<float> result(v1.size());

    for (int i = 0; i < v1.size(); i++) {
        result[i] = v1[i] * v2[i];
    }

    return result;
}

vector<vector<float>> elementWiseMult(const vector<vector<float>>& matrix, const vector<float>& vec) {
    // Check that dimensions are compatible
    if (matrix.size() != vec.size()) {
        throw "Matrix rows and vector size must match for element-wise multiplication";
    }

    vector<vector<float>> result(matrix.size(), vector<float>(matrix[0].size()));

    for (int i = 0; i < matrix.size(); ++i) {
        // Check that each row of the matrix has the same number of columns
        if (matrix[i].size() != result[i].size()) {
            throw "Matrix dimensions must be consistent for element-wise multiplication";
        }

        for (int j = 0; j < matrix[i].size(); ++j) {
            result[i][j] = matrix[i][j] * vec[i];
        }
    }

    return result;
}

vector<vector<float>> elementWiseMult(const vector<float>& vec, const vector<vector<float>>& matrix) {
    // Check that dimensions are compatible
    if (vec.size() != matrix.size()) {
        throw "Vector size and matrix rows must match for element-wise multiplication";
    }

    vector<vector<float>> result(matrix.size(), vector<float>(matrix[0].size()));

    for (int i = 0; i < matrix.size(); ++i) {
        // Check that each row of the matrix has the same number of columns
        if (matrix[i].size() != result[i].size()) {
            throw "Matrix dimensions must be consistent for element-wise multiplication";
        }

        for (int j = 0; j < matrix[i].size(); ++j) {
            result[i][j] = matrix[i][j] * vec[i];
        }
    }

    return result;
}

vector<vector<float>> elementWiseMult(const vector<vector<float>>& matrix1, const vector<vector<float>>& matrix2) {
    // Check that dimensions are compatible
    if (matrix1.size() != matrix2.size() || matrix1[0].size() != matrix2[0].size()) {
        throw "Matrix dimensions must match for element-wise multiplication";
    }

    vector<vector<float>> result(matrix1.size(), vector<float>(matrix1[0].size()));

    for (int i = 0; i < matrix1.size(); ++i) {
        for (int j = 0; j < matrix1[i].size(); ++j) {
            result[i][j] = matrix1[i][j] * matrix2[i][j];
        }
    }

    return result;
}



// Multiply a vector by a float 
vector<float> scaleVector(vector<float> v, float scale) {

    // Create a result vector
    vector<float> result(v.size());

    // Multiply each element by scale
    for (int i = 0; i < v.size(); i++) {
        result[i] = v[i] * scale;
    }

    return result;
}

// Multiply a matrix by a scalar 
vector<vector<float>> scaleMatrix(vector<vector<float>> mat, float scale) {

    vector<vector<float>> result(mat.size(), vector<float>(mat[0].size()));

    for (int i = 0; i < mat.size(); i++) {
        for (int j = 0; j < mat[0].size(); j++) {
            result[i][j] = mat[i][j] * scale;
        }
    }

    return result;
}

vector<vector<float>> divideMatrixByScalar(const vector<vector<float>>& mat, float divisor) {
    vector<vector<float>> result(mat.size(), vector<float>(mat[0].size()));

    for (int i = 0; i < mat.size(); i++) {
        for (int j = 0; j < mat[0].size(); j++) {
            result[i][j] = mat[i][j] / divisor;
        }
    }

    return result;
}