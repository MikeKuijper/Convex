#pragma once
#include "Matrix.h"
#include <initializer_list>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <chrono>

//TO DO:
// * Error handling
// * GPU memory allocation
// * GPU matrix multiplication
// * GPU kernel
// * rename MatrixMultiplication methods
// * (opt.) integrate with Convex class

namespace ConvexGPU {
	int debugMode = 0; //2

	enum activationFunction : unsigned char {
		SIGMOID = 0, TANH, RELU, NONE
	};

	void testSetupIncrement();
	void testSetupMatrixMultiply();

	void matrixMultiplyGPU(Matrix <double> *a, Matrix <double> *b, Matrix <double> *output);

	class ImageClassDataset {
	public:
		std::vector <Matrix <double>> m_images;
		std::vector <Matrix <double>> m_imagesFlattened;
		std::vector <unsigned int> m_labels;
		uint32_t m_size;
		uint32_t m_rows;
		uint32_t m_columns;

		ImageClassDataset();
		ImageClassDataset(const char* _imagePath, const char* _labelPath, bool _swapEndianness);

		std::ifstream* deserialiseMNISTImages(std::ifstream* _inputStream, bool _swapEndianness = false);
		void deserialiseMNISTImages(const char* _path, bool _swapEndianness = false);
		//std::ifstream* deserialiseMNISTImagesMatrix(std::ifstream* _inputStream, bool _swapEndianness = false);
		//void deserialiseMNISTImagesMatrix(const char* _path, bool _swapEndianness = false);
		std::ifstream* deserialiseMNISTLabels(std::ifstream* _inputStream, bool _swapEndianness = false);
		void deserialiseMNISTLabels(const char* _path, bool _swapEndianness = false);

		void deserialiseMNIST(const char* _imagePath, const char* _labelPath, bool _swapEndianness = false);

		void flatten();
		//void flattenMatrix();

		void shuffle();
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

		std::vector <Matrix<double>*> m_weightMatrixes;
		std::vector <std::vector <double>> m_biasMatrixes;
		std::vector <Matrix<double>*> m_activations;
		std::vector <std::vector <double>> m_networkErrors;
		std::vector <Matrix<double>*> m_layerMatrixes;

		NeuralNetwork();
		NeuralNetwork(std::vector <int> _structure, hardwareMode _hardwareMode);
		NeuralNetwork(const char* _path);
		void generateNetwork();
		void initNetwork();
		Matrix<double>* feed(Matrix<double>* _input);
		Matrix<double>* feed(Matrix<double>* _input, int _rangeStart, int _rangeEnd);
		Matrix<double>* train(Matrix<double>* _input, Matrix<double>* _targetOutput);
		Matrix<double>* train(Matrix<double>* _input, Matrix<double>* _targetOutput, int _rangeStart, int _rangeEnd);
		double train(ConvexGPU::ImageClassDataset* _dataset, bool _assessPerformance = false);
		void trainSequence(ConvexGPU::ImageClassDataset* _dataset, int _epochs, const char* _path);
		double assess(ConvexGPU::ImageClassDataset* _dataset);

		void freeMemory();

		std::ofstream* serialise(std::ofstream* _outputStream, bool _swapEndianness = false);
		void serialise(const char * _path, bool _swapEndianness = false);

		std::ifstream* deserialise(std::ifstream* _inputStream, bool _swapEndianness = false);
		void deserialise(const char* _path, bool _swapEndianness = false);

		double normalise(double _input);
		double deriveNormalise(double _input);

		double* getWeight(int _n1Layer, int _n1Neuron, int _n2Layer, int _n2Neuron);

		void matrixMultiplyCPU(Matrix<double>* _matrixA, Matrix<double>* _matrixB, Matrix<double>* output);
	};

	class AutoEncoder {
	public:
		unsigned int m_commonLayer;
		double m_learningRate;
		hardwareMode m_hardwareMode;
		activationFunction m_activationFunction;
		std::vector <int> m_fullNetworkStructure;
		std::vector <int> m_encoderNetworkStructure;
		std::vector <int> m_decoderNetworkStructure;

		NeuralNetwork m_network;
		NeuralNetwork* getEncoderNetwork();
		NeuralNetwork* getDecoderNetwork();

		AutoEncoder();
		AutoEncoder(std::vector <int> _structure, hardwareMode _hardwareMode);
		AutoEncoder(std::vector <int> _structure, int _commonLayer, hardwareMode _hardwareMode);

		void generateNetworks();

		Matrix<double>* encode(ConvexGPU::Matrix<double>* _input);
		Matrix<double>* decode(ConvexGPU::Matrix<double>* _input);

		Matrix<double>* feed(ConvexGPU::Matrix<double>* _input);
		double train(ConvexGPU::Matrix<double>* _input);

		std::ofstream* serialise(std::ofstream* _outputStream, bool _swapEndianness = false);
		void serialise(const char * _path);

		std::ifstream* deserialise(std::ifstream* _inputStream, bool _swapEndianness = false);
		void deserialise(const char* _path);
	};
}

template <typename Function>
double timeExecution(Function f, const char* text) {
	auto start = std::chrono::high_resolution_clock::now();
	f();
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
	std::cout << "[CONVEX] " << text << ": " << duration.count() / 1000.0f << " ms" << std::endl;

	return (double)duration.count();
}

template <typename Function>
double timeExecution(Function f, const char* text, int enable) {
	if (enable <= ConvexGPU::debugMode)
	{
		return timeExecution(f, text);
	}
	else {
		f();
		return NULL;
	}
}