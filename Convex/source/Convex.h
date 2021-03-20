#pragma once
#include <vector>
#include <iostream>
#include <random>
#include <algorithm>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include <chrono>

//TO DO:
// * Error handling
// * GPU implementation
// * Neuroevolution
// * GANs
// * Using template literals for the serialisation
// * Revise serialisation headers

// * Parallel processing in applying activationFunction and adding biases

template <typename Function>
double timeExecution(Function f, const char* text) {
	auto start = std::chrono::high_resolution_clock::now();
	f();
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
	std::cout << "[CONVEX] " << text << ": " << duration.count() / 1000.0f << " ms" << std::endl;

	return duration.count();
}

namespace Convex {
	//char VERSION[16] = "v1.0.0a";

	enum activationFunction : unsigned char {
		SIGMOID = 0, TANH, RELU, NONE
	};

	enum hardwareMode : unsigned char {
		CPU = 0, CPU_MT, GPU_CUDA
	};

	class ImageClassDataset {
	public:
		std::vector <std::vector<std::vector<double>>> m_images;
		std::vector <std::vector<double>> m_imagesFlattened;
		std::vector <int> m_labels;
		uint32_t m_size;
		uint32_t m_rows;
		uint32_t m_columns;

		ImageClassDataset();
		ImageClassDataset(const char* _imagePath, const char* _labelPath, bool _swapEndianness);

		std::ifstream* deserialiseMNISTImages(std::ifstream* _inputStream, bool _swapEndianness = false);
		void deserialiseMNISTImages(const char* _path, bool _swapEndianness = false);
		std::ifstream* deserialiseMNISTLabels(std::ifstream* _inputStream, bool _swapEndianness = false);
		void deserialiseMNISTLabels(const char* _path, bool _swapEndianness = false);

		void deserialiseMNIST(const char* _imagePath, const char* _labelPath, bool _swapEndianness = false);

		void flatten();
	};

	class NeuralNetwork {
	public:
		int m_networkLength;
		std::vector <int> m_networkStructure;
		double m_learningRate;
		double m_cost;
		double m_score;
		double m_globalError;

		activationFunction m_activationFunction;
		hardwareMode m_hardwareMode;

		std::vector <std::vector <std::vector <double>>> m_weightMatrixes;
		std::vector <std::vector <double>> m_biasMatrixes;
		std::vector <std::vector <double>> m_activations;
		std::vector <std::vector <double>> m_networkErrors;

		NeuralNetwork();
		NeuralNetwork(std::vector <int> _structure);
		NeuralNetwork(const char* _path);
		void generateNetwork();
		std::vector <double> feed(std::vector <double>* _input);
		std::vector <double> feed(std::vector <double>* _input, int _rangeStart, int _rangeEnd);
		std::vector <double> train(std::vector <double>* _input, std::vector <double>* _targetOutput);
		std::vector <double> train(std::vector <double>* _input, std::vector <double>* _targetOutput, int _rangeStart, int _rangeEnd);
		double train(ImageClassDataset* _dataset);
		void trainSequence(ImageClassDataset* _dataset, int _epochs, const char* _path);
		double assess(ImageClassDataset* _dataset);

		std::vector <std::vector <double>> matrixMultiply(std::vector<std::vector <double>>* _matrixA, std::vector<std::vector <double>>* _matrixB);

		std::ofstream* serialise(std::ofstream* _outputStream, bool _swapEndianness = false);
		void serialise(const char * _path, bool _swapEndianness = false);

		std::ifstream* deserialise(std::ifstream* _inputStream, bool _swapEndianness = false);
		void deserialise(const char* _path, bool _swapEndianness = false);

		double normalise(double _input);
		double deriveNormalise(double _input);

		double* getWeight(int _n1Layer, int _n1Neuron, int _n2Layer, int _n2Neuron);
		double* getWeight(int _n1Layer, int _n1Neuron, int _n2Layer, int _n2Neuron, std::vector<std::vector<std::vector<double>>>* _ptrWeightMatrixes);
	};

	class AutoEncoder {
	public:
		int m_sharedLayer;
		double m_learningRate;
		activationFunction m_activationFunction;
		std::vector <int> m_fullNetworkStructure;
		std::vector <int> m_encoderNetworkStructure;
		std::vector <int> m_decoderNetworkStructure;

		NeuralNetwork m_network;
		NeuralNetwork getEncoderNetwork();
		NeuralNetwork getDecoderNetwork();

		AutoEncoder();
		AutoEncoder(std::vector <int> _structure);
		AutoEncoder(std::vector <int> _structure, int _sharedLayer);

		void generateNetworks();

		std::vector <double> encode(std::vector <double>* _input);
		std::vector <double> decode(std::vector <double>* _input);

		std::vector <double> feed(std::vector <double>* _input);
		double train(std::vector <double>* _input);

		std::ofstream* serialise(std::ofstream* _outputStream);
		void serialise(const char * _path);

		std::ifstream* deserialise(std::ifstream* _inputStream);
		void deserialise(const char* _path);
	};

	double randomNumber(double _min, double _max);

	template <class variableType>
	void writeVariable(variableType* _variable, std::ofstream* _fs, bool _swapEndianness = false);
	template <class variableType>
	variableType readVariable(variableType* _variable, std::ifstream* _fs, bool _swapEndianness = false);

	template <class number>
	void writeVector(std::vector <number>* _vector, std::ofstream* _fs, bool _swapEndianness = false);
	template <class number>
	void writeVector(std::vector <std::vector <number>>* _vector, std::ofstream* _fs, bool _swapEndianness = false);
	template <class number>
	void writeVector(std::vector <std::vector <std::vector <number>>>* _vector, std::ofstream* _fs, bool _swapEndianness = false);

	template <typename number>
	void readVector(std::vector <number>* _vector, std::ifstream* _fs);
	template <typename number>
	void readVector(std::vector <std::vector <number>>* _vector, std::ifstream* _fs);

	template <class number>
	std::vector<number> subVector(std::vector <number>* _vector, int _i, int _j);

	template <class variable>
	void endianSwap(variable *objp);

	template <typename T>
	std::vector<T> flattenVector(const std::vector<std::vector<T>>& v);
}
