#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <random>
#include <cmath>
#include <iomanip>
#include <tuple>
#include "artificial_neural_network.cpp" // Include your neural network implementation

using namespace std;

// Define a structure to hold the dataset
struct Dataset {
    vector<vector<float>> data;
    vector<string> labels;
};

// Function to convert string to float
float stringToFloat(const string& str) {
    istringstream iss(str);
    float f;
    iss >> f;
    return f;
}

// Function to import dataset
Dataset importDataset(const string& file_dir) {
    Dataset dataset;
    ifstream file(file_dir);
    string line;
    if (file.is_open()) {
        // Read the first line to get labels
        getline(file, line);
        istringstream iss(line);
        string label;
        while (getline(iss, label, ',')) {
            // Remove any leading/trailing whitespaces
            label.erase(remove_if(label.begin(), label.end(), ::isspace), label.end());
            dataset.labels.push_back(label);
        }

        // Read the rest of the lines to get data
        while (getline(file, line)) {
            vector<float> row;
            istringstream iss(line);
            string value;
            while (getline(iss, value, ',')) {
                // Remove any leading/trailing whitespaces
                value.erase(remove_if(value.begin(), value.end(), ::isspace), value.end());
                // Convert string to float
                row.push_back(stringToFloat(value));
            }
            dataset.data.push_back(row);
        }
        file.close();
    }
    else {
        cerr << "Unable to open file: " << file_dir << endl;
    }
    return dataset;
}
// Function to shuffle dataset
void shuffleDataset(Dataset& dataset) {
    random_device rd;
    mt19937 g(rd());
    shuffle(dataset.data.begin(), dataset.data.end(), g);
}

// Function to split dataset into attributes and class labels
pair<vector<vector<float>>, vector<float>> splitDataset(const Dataset& dataset) {
    vector<vector<float>> x_dataset;
    vector<float> y_dataset;
    for (const auto& row : dataset.data) {
        vector<float> x_row(row.begin(), row.end() - 1);
        x_dataset.push_back(x_row);
        y_dataset.push_back(row.back());
    }
    return { x_dataset, y_dataset };
}

// Function to normalize dataset
pair<vector<vector<float>>, vector<vector<float>>> normalize(
    const vector<vector<float>>& x_train,
    const vector<vector<float>>& x_test) {

    vector<vector<float>> norm_train;
    vector<vector<float>> norm_test;

    vector<float> max_vals(x_train[0].size(), -numeric_limits<float>::infinity());
    vector<float> min_vals(x_train[0].size(), numeric_limits<float>::infinity());

    // Find max and min values for each feature in the training set
    for (const auto& d : x_train) {
        for (size_t i = 0; i < d.size(); ++i) {
            if (!isnan(d[i])) {
                max_vals[i] = max(max_vals[i], d[i]);
                min_vals[i] = min(min_vals[i], d[i]);
            }
        }
    }

    // Normalize training set
    for (const auto& d : x_train) {
        vector<float> data_vals;
        for (size_t i = 0; i < d.size(); ++i) {
            if (isnan(d[i])) {
                data_vals.push_back(0.0); // Replace missing values with 0
            }
            else {
                float val = d[i];
                data_vals.push_back((val - min_vals[i]) / (max_vals[i] - min_vals[i]));
            }
        }
        norm_train.push_back(data_vals);
    }

    // Normalize test set
    for (const auto& d : x_test) {
        vector<float> data_vals;
        for (size_t i = 0; i < d.size(); ++i) {
            if (isnan(d[i])) {
                data_vals.push_back(0.0); // Replace missing values with 0
            }
            else {
                float val = d[i];
                data_vals.push_back((val - min_vals[i]) / (max_vals[i] - min_vals[i]));
            }
        }
        norm_test.push_back(data_vals);
    }

    return { norm_train, norm_test };
}

// Function to split dataset into training and testing sets
tuple<vector<vector<float>>, vector<vector<float>>,
vector<float>, vector<float>> randomPartition(
        vector<vector<float>>& x_dataset,
        vector<float>& y_dataset) {

    vector<vector<float>> x_train_set_80, x_test_set_20;
    vector<float> y_train_set_80, y_test_set_20;

    random_device rd;
    mt19937 g(rd());
    shuffle(x_dataset.begin(), x_dataset.end(), g);
    shuffle(y_dataset.begin(), y_dataset.end(), g);

    size_t train_size = x_dataset.size() * 0.8;

    x_train_set_80.assign(x_dataset.begin(), x_dataset.begin() + train_size);
    y_train_set_80.assign(y_dataset.begin(), y_dataset.begin() + train_size);
    x_test_set_20.assign(x_dataset.begin() + train_size, x_dataset.end());
    y_test_set_20.assign(y_dataset.begin() + train_size, y_dataset.end());

    return { x_train_set_80, x_test_set_20, y_train_set_80, y_test_set_20 };
}

// Function to calculate accuracy
// Function to calculate accuracy
// Function to calculate accuracy
// Function to calculate accuracy
double calculateAccuracy(const vector<float>& y_data, const vector<float>& y_pred_data) {
    size_t correct = 0;
    for (size_t i = 0; i < y_data.size(); ++i) {
        int predicted_class = round(y_pred_data[i]);
        if (y_data[i] == predicted_class) {
            correct++;
        }
    }
    return static_cast<double>(correct) / y_data.size() * 100.0;
}



int main() {
    // Importing the dataset
    Dataset dataset = importDataset("diabetes.csv");

    // Shuffle the dataset
    shuffleDataset(dataset);

    // Split dataset into attributes and class labels
    pair<vector<vector<float>>, vector<float>> xy_dataset = splitDataset(dataset);
    vector<vector<float>> x_dataset = xy_dataset.first;
    vector<float> y_dataset = xy_dataset.second;
    // Split dataset into training and testing sets
    tuple<vector<vector<float>>, vector<vector<float>>,
        vector<float>, vector<float>> xy_train_test = randomPartition(x_dataset, y_dataset);
    vector<vector<float>> x_train = get<0>(xy_train_test);
    vector<vector<float>> x_test = get<1>(xy_train_test);
    vector<float> y_train = get<2>(xy_train_test);
    vector<float> y_test = get<3>(xy_train_test);
    // Normalize dataset
    pair<vector<vector<float>>, vector<vector<float>>> norm_x = normalize(x_train, x_test);
    vector<vector<float>> norm_x_train = norm_x.first;
    vector<vector<float>> norm_x_test = norm_x.second;
    // Convert y_train and y_test to 2D vectors
    vector<vector<float>> y_train_2d(y_train.size(), vector<float>(1));
    vector<vector<float>> y_test_2d(y_test.size(), vector<float>(1));
    for (size_t i = 0; i < y_train.size(); ++i) {
        y_train_2d[i][0] = y_train[i];
    }
    for (size_t i = 0; i < y_test.size(); ++i) {
        y_test_2d[i][0] = y_test[i];
    }
    float lambda = 0.00001;
    float alpha = 0.01;
    vector<int> neuronLayer = { static_cast<int>(norm_x_train[0].size()), 20, 10, 20, 1 }; // Update the neuron layers based on your dataset
    int max_iterations = 1000;
    ArtificialNeuralNetwork ann(lambda, alpha, neuronLayer, max_iterations);
    vector<vector<float>> y_pred_train_2d = ann.backProp(norm_x_train, y_train_2d);

    // Predict using ANN
    vector<vector<float>> y_pred_test_2d = ann.predict(norm_x_test);
    vector<float> y_pred_test;
    for (const auto& pred : y_pred_test_2d) {
        y_pred_test.push_back(round(pred[0]));
    }

   
    // Calculate accuracy
    double accuracy = calculateAccuracy(y_test, y_pred_test);
    cout << "Accuracy: " << fixed << setprecision(2) << accuracy << "%" << endl;

    return 0;
}
