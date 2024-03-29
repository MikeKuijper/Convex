#include "ConvexGPU.h"

namespace ConvexGPU {
	AutoEncoder::AutoEncoder() {}

	AutoEncoder::AutoEncoder(std::vector <int> _structure, hardwareMode m_hardwareMode) {
		m_fullNetworkStructure = _structure;
		m_commonLayer = std::floor(m_fullNetworkStructure.size() / 2);

		m_learningRate = 1;
		m_activationFunction = SIGMOID;

		generateNetworks();
	}

	AutoEncoder::AutoEncoder(std::vector <int> _structure, int _sharedLayer, hardwareMode _hardwareMode) {
		m_fullNetworkStructure = _structure;
		m_commonLayer = _sharedLayer;
		m_hardwareMode = _hardwareMode;

		m_learningRate = 1;
		m_activationFunction = SIGMOID;

		generateNetworks();
	}

	void AutoEncoder::generateNetworks() {
		m_encoderNetworkStructure = subVector(&m_fullNetworkStructure, 0, m_commonLayer);
		m_decoderNetworkStructure = subVector(&m_fullNetworkStructure, m_commonLayer, m_fullNetworkStructure.size() - 1);

		m_network = NeuralNetwork(m_fullNetworkStructure, m_hardwareMode);
		m_network.m_activationFunction = this->m_activationFunction;
		m_network.m_learningRate = this->m_learningRate;
	}

	ConvexGPU::Matrix<double>* AutoEncoder::encode(ConvexGPU::Matrix<double>* _input) {
		return m_network.feed(_input, 0, m_commonLayer);
	}

	ConvexGPU::Matrix<double>* AutoEncoder::decode(ConvexGPU::Matrix<double>* _input) {
		return m_network.feed(_input, m_commonLayer, m_fullNetworkStructure.size() - 1);
	}

	ConvexGPU::Matrix<double>* AutoEncoder::feed(ConvexGPU::Matrix<double>* _input) {
		return m_network.feed(_input);
	}

	NeuralNetwork* AutoEncoder::getEncoderNetwork() {
		NeuralNetwork encoder;
		encoder.m_weightMatrixes = subVector(&m_network.m_weightMatrixes, 0, m_commonLayer - 1);
		encoder.m_biasMatrixes = subVector(&m_network.m_biasMatrixes, 0, m_commonLayer);
		encoder.m_networkErrors = subVector(&m_network.m_networkErrors, 0, m_commonLayer);
		encoder.m_activationFunction = m_activationFunction;
		return &encoder;
	}

	NeuralNetwork* AutoEncoder::getDecoderNetwork() {
		NeuralNetwork decoder;
		decoder.m_weightMatrixes = subVector(&m_network.m_weightMatrixes, m_commonLayer, m_network.m_weightMatrixes.size() - 1);
		decoder.m_biasMatrixes = subVector(&m_network.m_biasMatrixes, m_commonLayer - 1, m_network.m_biasMatrixes.size() - 1);
		decoder.m_networkErrors = subVector(&m_network.m_networkErrors, m_commonLayer - 1, m_network.m_networkErrors.size() - 1);
		decoder.m_activationFunction = m_activationFunction;
		return &decoder;
	}

	double AutoEncoder::train(ConvexGPU::Matrix<double>* _input) {
		m_network.m_learningRate = m_learningRate;
		m_network.m_activationFunction = m_activationFunction;

		//m_network.train(_input, _input);
		return m_network.m_cost;
	}

	std::ofstream* AutoEncoder::serialise(std::ofstream* _outputStream, bool _swapEndianness) {
		writeVariable(&m_commonLayer, _outputStream, _swapEndianness);
		writeVariable(&m_learningRate, _outputStream, _swapEndianness);
		writeVariable(&m_activationFunction, _outputStream, _swapEndianness);
		writeVector(&m_fullNetworkStructure, _outputStream, _swapEndianness);

		m_network.serialise(_outputStream);

		return _outputStream;
	}

	void AutoEncoder::serialise(const char * _path) {
		std::ofstream file;
		file.open(_path, std::ios::binary);

		serialise(&file);
		file.close();
	}

	std::ifstream* AutoEncoder::deserialise(std::ifstream* _inputStream, bool _swapEndianness) {
		readVariable(&m_commonLayer, _inputStream, _swapEndianness);
		readVariable(&m_learningRate, _inputStream, _swapEndianness);
		readVariable(&m_activationFunction, _inputStream, _swapEndianness);
		readVector(&m_fullNetworkStructure, _inputStream, _swapEndianness);

		m_network.deserialise(_inputStream, _swapEndianness);

		m_encoderNetworkStructure = subVector(&m_fullNetworkStructure, 0, m_commonLayer);
		m_decoderNetworkStructure = subVector(&m_fullNetworkStructure, m_commonLayer, m_fullNetworkStructure.size() - 1);

		return _inputStream;
	}

	void AutoEncoder::deserialise(const char* _path) {
		std::ifstream file;
		file.open(_path, std::ios::binary);

		deserialise(&file);
		file.close();
	}
}