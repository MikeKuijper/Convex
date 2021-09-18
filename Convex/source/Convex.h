#pragma once
#include "Common.h"
#include "Matrix.h"

//TO DO:
// * Error handling
// * GPU implementation
// * Neuroevolution
// * GANs
// * Using template literals for the serialisation
// * Revise serialisation headers

// * Parallel processing in applying activationFunction and adding biases

namespace Convex {
	enum activationFunction : unsigned char {
		SIGMOID = 0, TANH, RELU, NONE
	};

	enum hardwareMode : unsigned char {
		CUDA, CPU
	};

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
		double train(ImageClassDataset* _dataset, bool _assessPerformance = false);
		void trainSequence(ImageClassDataset* _dataset, int _epochs, const char* _path);
		double assess(ImageClassDataset* _dataset);

		std::vector <std::vector <double>> matrixMultiply(std::vector<std::vector <double>>* _matrixA, std::vector<std::vector <double>>* _matrixB);

		void freeMemory();

		std::ofstream* serialise(std::ofstream* _outputStream, bool _swapEndianness = false);
		void serialise(const char * _path, bool _swapEndianness = false);

		std::ifstream* deserialise(std::ifstream* _inputStream, bool _swapEndianness = false);
		void deserialise(const char* _path, bool _swapEndianness = false);

		double normalise(double _input);
		double deriveNormalise(double _input);

		double* getWeight(int _n1Layer, int _n1Neuron, int _n2Layer, int _n2Neuron);

		Matrix <double> matrixMultiply(Matrix <double>* _matrixA, Matrix <double>* _matrixB);
	};

	class AutoEncoder {
	public:
		unsigned int m_sharedLayer;
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

		std::ofstream* serialise(std::ofstream* _outputStream, bool _swapEndianness = false);
		void serialise(const char * _path);

		std::ifstream* deserialise(std::ifstream* _inputStream, bool _swapEndianness = false);
		void deserialise(const char* _path);
	};
}