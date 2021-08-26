#pragma once
#include "Common.h"

//TO DO:
// * Error handling
// * GPU implementation
// * Neuroevolution
// * GANs
// * Using template literals for the serialisation
// * Revise serialisation headers

// * Parallel processing in applying activationFunction and adding biases

namespace Convex {
	//char VERSION[16] = "v1.0.0a";

	int debugMode = 0; //2

	enum activationFunction : unsigned char {
		SIGMOID = 0, TANH, RELU, NONE
	};

	/*enum hardwareMode : unsigned char {
		CPU = 0, CPU_MT
	};*/

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

		std::vector <std::vector <std::vector <double>>> m_weightMatrixes;
		std::vector <std::vector <double>> m_biasMatrixes;
		std::vector <std::vector <double>> m_activations;
		std::vector <std::vector <double>> m_networkErrors;

		NeuralNetwork();
		NeuralNetwork(std::vector <int> _structure);
		NeuralNetwork(const char* _path);
		void generateNetwork();
		std::vector <double> feed(std::vector <double>* _input);
		std::vector <double> feed(std::vector <double>* _input, unsigned int _rangeStart, unsigned int _rangeEnd);
		std::vector <double> train(std::vector <double>* _input, std::vector <double>* _targetOutput);
		std::vector <double> train(std::vector <double>* _input, std::vector <double>* _targetOutput, int _rangeStart, int _rangeEnd);
		double train(Convex::ImageClassDataset* _dataset, bool _assessPerformance = false);
		void trainSequence(Convex::ImageClassDataset* _dataset, int _epochs, const char* _path);
		double assess(Convex::ImageClassDataset* _dataset);

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

template <typename Function>
double timeExecution(Function f, const char* text) {
	auto start = std::chrono::high_resolution_clock::now();
	f();
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
	std::cout << "[CONVEX] " << text << ": " << duration.count() / 1000.0f << " ms" << std::endl;

	return (double) duration.count();
}

template <typename Function>
double timeExecution(Function f, const char* text, int enable) {
	if (enable <= Convex::debugMode)
	{
		return timeExecution(f, text);
	}
	else {
		f();
		return NULL;
	}
}