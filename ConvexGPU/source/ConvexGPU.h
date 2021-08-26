#pragma once
#include "Convex.h"
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

using namespace Convex;

namespace ConvexGPU {
	void testSetupIncrement();
	void testSetupMatrixMultiply();

	Matrix <double> matrixMultiply2D(Matrix <double> *a, Matrix <double> *b);

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
		void flattenMatrix();

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
		//hardwareMode m_hardwareMode;

		std::vector <Matrix<double>> m_weightMatrixes;
		std::vector <std::vector <double>> m_biasMatrixes;
		std::vector <Matrix<double>> m_activations;
		std::vector <std::vector <double>> m_networkErrors;

		NeuralNetwork();
		NeuralNetwork(std::vector <int> _structure);
		NeuralNetwork(const char* _path);
		void generateNetwork();
		Matrix<double> feed(Matrix<double>* _input);
		Matrix<double> feed(Matrix<double>* _input, int _rangeStart, int _rangeEnd);
		Matrix<double> train(Matrix<double>* _input, Matrix<double>* _targetOutput);
		Matrix<double> train(Matrix<double>* _input, Matrix<double>* _targetOutput, int _rangeStart, int _rangeEnd);
		double train(ConvexGPU::ImageClassDataset* _dataset, bool _assessPerformance = false);
		void trainSequence(ConvexGPU::ImageClassDataset* _dataset, int _epochs, const char* _path);
		double assess(ConvexGPU::ImageClassDataset* _dataset);

		std::vector <std::vector <double>> matrixMultiply(std::vector<std::vector <double>>* _matrixA, std::vector<std::vector <double>>* _matrixB);

		void freeMemory();

		std::ofstream* serialise(std::ofstream* _outputStream, bool _swapEndianness = false);
		void serialise(const char * _path, bool _swapEndianness = false);

		std::ifstream* deserialise(std::ifstream* _inputStream, bool _swapEndianness = false);
		void deserialise(const char* _path, bool _swapEndianness = false);

		double normalise(double _input);
		double deriveNormalise(double _input);

		double* getWeight(int _n1Layer, int _n1Neuron, int _n2Layer, int _n2Neuron);
	};
}