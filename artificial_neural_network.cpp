// Artificial Neural Networks.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#pragma once

#include <iostream>
#include <vector>
#include <cmath>
#include "ann_functions.h"
#include <iomanip>
#include <cstdlib>

using namespace std;

class ArtificialNeuralNetwork {
public:
    // Constructor 
    ArtificialNeuralNetwork(float lambdaI, float alphaI, vector<int> neuronLayerI, int max_iterationsI) {
        lambda = lambdaI;
        alpha = alphaI;
        neuronLayer = neuronLayerI;
        layers = neuronLayerI.size() - 1;
        max_iterations = max_iterationsI;
        // Generate theta randomly based on the dimensions of neuronLayer
        theta.resize(layers);
        for (int i = 0; i < layers; i++) {
            int rows = neuronLayer[i + 1];
            int cols = neuronLayer[i] + 1;
            theta[i].resize(rows, vector<float>(cols));
            for (int j = 0; j < rows; j++) {
                for (int k = 0; k < cols; k++) {
                    theta[i][j][k] = static_cast<float>(rand()) / RAND_MAX;
                }
                // Set bias unit weight to zero for regularization
                theta[i][j][0] = 0.0f;
            }
        }
        
    }
    // Member variables
    float lambda;
    float alpha;
    vector<int> neuronLayer;
    int layers;
    vector<float> sigmoid(vector<float> z);

    vector<vector<vector<vector<float>>>> a_list;
    int max_iterations; // Maximum number of iterations
    vector<vector<vector<float>>> theta;

    // Member functions
    vector<vector<float>> forwardProp(vector<float> xi, vector<vector<vector<float>>> &a) {
        vector<vector<float>> ali;
        vector<vector<vector<float>>> z;
        ali.push_back({ 1.0f }); // bias value
        
        for (float val : xi) {
            ali.push_back({ val });
        }
        a.push_back(ali);
        for (int k = 1; k < layers; k++) {
            z.push_back(multiplyMatrices(theta[k - 1], a[k - 1]));
            a.push_back(g(z[k - 1])); // g func(z)
            a[k].insert(a[k].begin(), { 1.0f });
        }

        z.push_back(multiplyMatrices(theta[layers - 1], a[layers - 1]));
        a.push_back({ g_output(z[layers - 1]) }); // g func(z)
       
        return a[layers];
    }

    vector<vector<float>> backProp(vector<vector<float>> x, vector<vector<float>> y) {
        int n = x.size();
        
        vector<vector<vector<float>>> D(layers); // gradients of the weights of each layer
        // Store the predictions for each training instance
        vector<vector<float>> predictions;
        int iteration = 0;
        float previous_cost = numeric_limits<float>::max(); // Initialize with a large value
        float current_cost;
        do {
            predictions.clear();
            for (int i = 0; i < n; i++) {
               
                vector<vector<vector<float>>> a;
                vector<float> fx = transposeVector(forwardProp(x[i], a));
                predictions.push_back(fx); // Store the predicted output
                
                vector<vector<vector<float>>> delta(layers + 1);
                delta[layers].resize(1);
                delta[layers] = transposeVector(subtractVectors(fx, y[i])); // computes the delta values of all output neurons

                /*cout << "layers" << layers << endl;
                cout << "Running backpropagation" << endl;
                cout << "\n--------------------------------------------\n";
                cout << "Computing gradients based on training instance " << i + 1 << endl;*/
                for (int k = layers - 1; k >= 1; k--) { // computes the delta values of all neurons in the hidden layers
                    vector<vector<float>> transpose = transposeMatrix(theta[k]);
                    vector<vector<float>> thetaDelta = multiplyMatrices(transpose, delta[k + 1]);
                    vector<vector<float>>  thetaDeltaA = elementWiseMult(thetaDelta, a[k]);
                    vector<vector<float>> nA(a[k].size(), vector<float>(a[k][0].size()));

                    // Handle the bias term separately
                    for (int i = 0; i < a[k][0].size(); i++) {
                        nA[0][i] = 1.0f; // Set the bias term to 1.0
                    }
                    // Subtract each element from 1
                    for (int i = 1; i < a[k].size(); i++) {
                        for (int j = 0; j < a[k][i].size(); j++) {
                            nA[i][j] = 1.0 - a[k][i][j];
                        }
                    }
                    vector<vector<float>> thetaDeltaANa = elementWiseMult(thetaDeltaA, nA);
                    //cout << "deltaANa: [" << thetaDeltaANa[1][0] << "]" << endl;
                    delta[k].resize(thetaDeltaANa.size() - 1);
                    for (int j = 1; j <= thetaDeltaANa.size() - 1; j++) {
                        delta[k][j - 1] = thetaDeltaANa[j]; // Assign values to delta[k]
                    }
                }
               

                for (int k = layers - 1; k >= 0; k--) { // updates gradients of the weights of each layer, based on the current training instance
                    vector<vector<float>> deltaATranspose = multiplyMatrixMatrixTranspose(delta[k + 1], a[k]);
                    //cout << "layersError" << layers << endl;
                    // Compute the sum of D[0] and deltaATranspose
                    if (D[k].empty()) {
                        D[k] = deltaATranspose;
                    }
                    else {
                        D[k] = addMatrices(D[k], deltaATranspose);
                    }
                    // accumulates, in D(l=k), gradients computed based on the current training instance
                }
                
               
            }

            //cout << "The entire training set has been processed. Computing the average (regularized) gradients:\n";

            for (int k = layers - 1; k >= 0; k--) { // computes the final(regularized) gradients of the weights of each layer
                vector<vector<float>> Pk(theta[k].size(), vector<float>(theta[k][0].size(), 0.0)); // Create a matrix of zeros
                for (size_t i = 0; i < theta[k].size(); ++i) {
                    for (size_t j = 1; j < theta[k][i].size(); ++j) { // Start from 1 to exclude bias weights
                        Pk[i][j] = lambda * theta[k][i][j]; // Compute regularizer for non-bias weights
                    }
                }
                D[k] = divideMatrixByScalar((addMatrices(D[k], Pk)), n); // combines gradients w/ regularization terms; divides by #instances to obtain average gradient
            }

           

            // At this point, D(l=1) contains the gradients of the weights θ(l=1); (…); and D(l=L-1) contains the gradients of the weights θ(l=L-1)
            for (int k = layers - 1; k >= 0; k--) { // updates the weights of each layer based on their corresponding gradients
                theta[k] = subtractMatrices(theta[k], elementWiseMult(alpha, D[k]));
            }
            
            // Now Add the Big For Loop And S
            // Compute and display the regularized cost
            current_cost = computeRegularizedCost(y, predictions);

            cout << "Iteration " << iteration + 1 << ": Final (regularized) cost, J, based on the complete training set: " << current_cost << endl;

            // Check if the stopping criterion is met
            if (abs(current_cost - previous_cost) < 0.00001) {
                cout << "Stopping criterion met after " << iteration + 1 << " iterations." << endl;
                break;
            }

            previous_cost = current_cost;
            iteration++;

        } while (iteration < max_iterations);
        if (iteration == max_iterations) {
            cout << "Maximum number of iterations (" << max_iterations << ") reached." << endl;
        }
        return predictions;
    }

    // sigmoid
    vector<vector<float>> g(const vector<vector<float>>& z) {
        vector<vector<float>> result(z.size(), vector<float>(z[0].size()));

        for (size_t i = 0; i < z.size(); ++i) {
            for (size_t j = 0; j < z[0].size(); ++j) {
                result[i][j] = 1 / (1 + exp(-z[i][j]));
            }
        }

        return result;
        //return z;
    }
    // Sigmoid activation function for the output layer
    vector<vector<float>> g_output(const vector<vector<float>>& z) {
        vector<vector<float>> result(z.size(), vector<float>(z[0].size()));
        for (size_t i = 0; i < z.size(); ++i) {
            for (size_t j = 0; j < z[0].size(); ++j) {
                result[i][j] = 1 / (1 + exp(-z[i][j]));
            }
        }
        return result;
    }


    // After training, compute the regularized cost function
    // Cost Function J
    float costJ(float y, float fx) {
        return -y * log(fx) - (1 - y) * log(1 - fx);
    }
    // After training, compute the regularized cost function
    float computeRegularizedCost(const vector<vector<float>>& y, const vector<vector<float>>& predictions) {
        int n = y.size();
        int layers = predictions[0].size(); // Assuming each prediction corresponds to a layer
        // Step 1: J=0 // initializes the variable that will accumulate the total cost/error incurred by the network
        float J = 0.0;
        // Step 2: For each training instance (x(i), y(i))
        for (int i = 0; i < n; ++i) {
            float J_i = 0.0; // Initialize the J(i) value for the current instance
            // Step 2.1: Propagate x(i) and compute each of the network’s outputs, fθ(x(i))
            // This step is already done, and the outputs are stored in predictions[i]
            // Step 2.2: Compute J(i) = -y(i) .* log(fθ(x(i))) - (1-y(i)) .* log(1 - fθ(x(i)))
            for (int j = 0; j < layers; ++j) {
                J_i += costJ(y[i][j], predictions[i][j]);
            }
            // Accumulate the cost associated with instance i
            J += J_i;
            // Print the cost associated with each instance
            //cout << "Cost, J, associated with instance " << i + 1 << ": " << J_i << endl;
        }
        // Step 3: J = J / n // divides the total error/cost of the network by the number of training instances
        J /= n;
        // Step 4: S = computes the square of all weights of the network (except bias weights) and adds them up
        float S = 0;
        for (int k = 0; k < theta.size(); ++k) {
            for (size_t m = 0; m < theta[k].size(); ++m) {
                for (size_t p = 1; p < theta[k][m].size(); ++p) {  // Start from 1 to exclude bias weights
                    S += pow(theta[k][m][p], 2);
                }
            }
        }

        // Step 5: S = (λ/(2n)) * S // computes the term used to regularize the network’s cost
        S = (lambda / (2 * n)) * S;
        // Step 6: Return the regularized error/cost of the network, J+S, with respect to the entire training set
        return J + S;
    }



    vector<vector<float>> predict(vector<vector<float>> x) {
        vector<vector<float>> predictions;
        for (vector<float> xi : x) {
            vector<vector<vector<float>>> a;
            vector<float> prediction = transposeVector(forwardProp(xi, a));
            predictions.push_back(prediction);
        }
        return predictions;
    }
};

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu