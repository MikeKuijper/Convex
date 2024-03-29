#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "Common.h"
#include "ConvexGPU.h"

namespace ConvexGPU {
	NeuralNetwork::NeuralNetwork() {}

	NeuralNetwork::NeuralNetwork(std::vector <int> _structure, hardwareMode m_hardwareMode) {
		m_networkStructure = _structure;
		m_networkLength = _structure.size();
		m_learningRate = 0.1;
		m_activationFunction = SIGMOID;
		m_hardwareMode = CPU;

		generateNetwork();
		initNetwork();
	}

	NeuralNetwork::NeuralNetwork(const char* _path) {
		deserialise(_path);
	}

	void NeuralNetwork::generateNetwork() {
		m_weightMatrixes.clear();
		m_biasMatrixes.clear();
		m_networkErrors.clear();

		for (int i = 1; i < m_networkLength; i++) {
			Matrix <double>* weightMatrix = new Matrix<double>(m_networkStructure.at(i - 1), m_networkStructure.at(i));
			weightMatrix->m_hardwareMode = m_hardwareMode;

			for (int i = 0; i < weightMatrix->dimensions[1]; i++) {
				for (int j = 0; j < weightMatrix->dimensions[0]; j++) {
					weightMatrix->at2D(i, j) = randomNumber(-1.0f, 1.0f);
				}
			}

			m_weightMatrixes.push_back(weightMatrix);
		}

		for (int i = 0; i < m_networkLength; i++) {
			std::vector <double> biasMatrix;
			std::vector <double> errorMatrix;
			for (int j = 0; j < m_networkStructure.at(i); j++) {
				biasMatrix.push_back(randomNumber(-1.0f, 1.0f));
				errorMatrix.push_back(0);
			}
			m_biasMatrixes.push_back(biasMatrix);
			m_networkErrors.push_back(errorMatrix);
		}
	}

	void NeuralNetwork::initNetwork() {
		for (int i = 1; i < m_networkStructure.size(); i++) {
			Matrix <double>* result = new Matrix<double>(1, m_networkStructure[i]);
			result->m_hardwareMode = m_hardwareMode;
			m_layerMatrixes.push_back(result);
		}
	}

	Matrix<double>* NeuralNetwork::feed(Matrix<double>* _input) {
		return feed(_input, 0, m_weightMatrixes.size());
	}

	//TODO Write normalisation kernel and implement biases in previous layer
	Matrix<double>* NeuralNetwork::feed(Matrix<double>* _input, int _rangeStart, int _rangeEnd) {
		m_activations.clear();
		m_activations.push_back(_input);

		for (int i = _rangeStart; i < _rangeEnd; i++) {
			Matrix<double>* result = m_layerMatrixes.at(i);
			result->m_hardwareMode = m_hardwareMode;
			result->fill(0);

			if (m_hardwareMode == CPU) {
				matrixMultiplyCPU(m_activations[m_activations.size() - 1], m_weightMatrixes[i], result);
			}
			else if (m_hardwareMode == CUDA) {
				m_activations[m_activations.size() - 1]->copyMemoryToDevice();
				m_weightMatrixes[i]->copyMemoryToDevice();

				matrixMultiplyGPU(m_activations[m_activations.size() - 1], m_weightMatrixes[i], result);
				//result->copyMemoryToHost();
			}

			for (int j = 0; j < result->dimensions.at(1); j++) {
				result->at2D(0, j) = normalise(result->at2D(0, j) + m_biasMatrixes.at(i + 1).at(j));
			}

			//if (m_hardwareMode == CUDA) result->copyMemoryToDevice();

			m_activations.push_back(result);
		}

		return m_activations[m_activations.size() - 1];
	}

	Matrix<double>* NeuralNetwork::train(Matrix<double>* _input, Matrix<double>* _targetOutput) {
		return train(_input, _targetOutput, 0, m_networkLength - 1);
	}

	//TODO Check if this still works (cost reduces)
	Matrix<double>* NeuralNetwork::train(Matrix<double>* _input, Matrix<double>* _targetOutput, int _rangeStart, int _rangeEnd) {
		Matrix<double>* feedResult = feed(_input, _rangeStart, _rangeEnd);
		m_cost = 0;
		m_globalError = 0;

		for (int i = 0; i < feedResult->size(); i++) {
			double gradient = -deriveNormalise(feedResult->at2D(0, i));
			double error = feedResult->at2D(0, i) - _targetOutput->at2D(0, i);

			m_networkErrors.at(_rangeEnd).at(i) = error * gradient;
			m_cost += error;
		}

		if (m_learningRate != 0) {
			for (int i = _rangeEnd - 1; i >= _rangeStart; i--) {
				std::vector <double> nextLayerErrors = m_networkErrors.at(i + 1);

				//if(m_hardwareMode == CUDA) m_weightMatrixes.at(i)->copyMemoryToHost();

				for (int j = 0; j < m_networkStructure.at(i); j++) {
					double sum = 0;

					for (int k = 0; k < nextLayerErrors.size(); k++) {
						//double* weight = &m_weightMatrixes.at(i).at2D(j, k); // OF INTEREST
						double* weight = &m_weightMatrixes.at(i)->at2D(k, j); // OF INTEREST
						*weight += m_learningRate * nextLayerErrors.at(k) * m_activations.at(i)->at2D(0, j);

						sum += *weight * nextLayerErrors.at(k);
					}

					double currentError = sum * deriveNormalise(m_activations.at(i)->at2D(0, j));
					m_networkErrors.at(i).at(j) = currentError;
					m_globalError += abs(currentError);
					m_biasMatrixes.at(i).at(j) += m_learningRate * currentError;
				}

				if(m_hardwareMode == CUDA) m_weightMatrixes.at(i)->copyMemoryToDevice();
			}
		}
		return feedResult;
	}

	double NeuralNetwork::train(ConvexGPU::ImageClassDataset* _dataset, bool _assessPerformance) {
		double cost = 0;
		double score = 0;
		if (_assessPerformance == true) {
			for (int n = 0; n < _dataset->m_size; n++) {
				Matrix <double> targetResult(1, 10);
				targetResult.m_hardwareMode = CPU;
				targetResult.fill(0);
				targetResult[_dataset->m_labels[n]] = 1;

				Matrix<double>* feedResult = train(&_dataset->m_imagesFlattened[n], &targetResult);

				targetResult.freeMemory();

				cost += m_cost / _dataset->m_size;

				//TODO: Solve with iterators
				unsigned int maxElementIndex = feedResult->at(0);
				double maxElement = 0;
				for (int i = 0; i < feedResult->dimensions[1]; i++) {
					if (feedResult->at(i) > maxElement) {
						maxElement = feedResult->at(i);
						maxElementIndex = i;
					}
				}

				//std::vector<double>::iterator maxElement = std::max_element(feedResult.begin(), feedResult.end());
				//int maxElementIndex = std::distance(feedResult.begin(), maxElement);

				if (maxElementIndex == _dataset->m_labels[n]) {
					score++;
				}
			}

			this->m_score = score / _dataset->m_size;
			return cost;
		}
		else {
			for (int n = 0; n < _dataset->m_size; n++) {
				Matrix<double> targetResult(1, 10);
				targetResult.fill(0);
				targetResult[_dataset->m_labels[n]] = 1;

				Matrix<double>* feedResult = train(&_dataset->m_imagesFlattened[n], &targetResult);
				if (m_hardwareMode == CUDA) targetResult.freeMemoryGPU();
				else if (m_hardwareMode == CPU) targetResult.freeMemoryCPU();
				cost += m_cost / _dataset->m_size;
			}
			return cost;
		}
	}

	void NeuralNetwork::trainSequence(ConvexGPU::ImageClassDataset* _dataset, int _epochs, const char* _path) {
		std::cout << "[CONVEX] Training start" << std::endl;

		double minCost = assess(_dataset);
		int nonImprovementCount = 0;

		for (int n = 0; n < _epochs; n++) {
			//if (n % 2 == 0) m_hardwareMode = ConvexGPU::CPU;
			//else m_hardwareMode = ConvexGPU::CUDA;
			double cost = train(_dataset, n % 1 == 0);
			std::cout << "[CONVEX] Train epoch " << n << ": " << cost << " (" << round((float)m_score * 10000) / 100 << "%)" << std::endl;

			if (cost < minCost) {
				//serialise(_path);
				minCost = cost;
				nonImprovementCount = 0;
			}
			else {
				nonImprovementCount++;
				if (nonImprovementCount == 16) {
					double newLearningRate = m_learningRate * 0.90;
					//deserialise(_path);
					m_learningRate = newLearningRate;

					nonImprovementCount = 0;
					std::cout << "[CONVEX] Adjusting learning rate to " << m_learningRate << std::endl;
				}
			}
		}
	}

	double NeuralNetwork::assess(ConvexGPU::ImageClassDataset* _dataset) {
		std::cout << "[CONVEX] Assessing network...";

		double learningRateTemp = m_learningRate;
		m_learningRate = 0;

		double cost = train(_dataset);
		m_learningRate = learningRateTemp;

		std::cout << " done" << std::endl;
		std::cout << "[CONVEX] Cost: " << cost << " (" << round((float)m_score * 10000) / 100 << "%)" << std::endl;
		return cost;
	}

	template <typename T>
	std::vector<T> flattenVector(const std::vector<std::vector<T>>& v) {
		std::size_t total_size = 0;
		for (const auto& sub : v)
			total_size += sub.size();
		std::vector<T> result;
		result.reserve(total_size);
		for (const auto& sub : v)
			result.insert(result.end(), sub.begin(), sub.end());
		return result;
	}

	void NeuralNetwork::freeMemory() {
		for (int i = 0; i < m_activations.size() - 2; i++) {
			if (m_hardwareMode == CUDA) m_activations[i]->freeMemoryGPU();
			else if (m_hardwareMode == CPU) m_activations[i]->freeMemoryCPU();
		}
	}

	std::ofstream* NeuralNetwork::serialise(std::ofstream* _stream, bool _swapEndianness) {
		writeVariable(&m_learningRate, _stream, _swapEndianness);
		writeVariable(&m_activationFunction, _stream, _swapEndianness);
		//writeVariable(&m_hardwareMode, _stream, _swapEndianness);

		writeVector(&m_networkStructure, _stream, _swapEndianness);
		writeVector(&m_biasMatrixes, _stream, _swapEndianness);
		writeMatrixVector(&m_weightMatrixes, _stream, _swapEndianness);

		return _stream;
	}
	void NeuralNetwork::serialise(const char * _path, bool _swapEndianness) {
		std::cout << "[CONVEX] Saving network to " << _path << "...";

		std::ofstream file;
		file.open(_path, std::ios::binary);

		serialise(&file, _swapEndianness);

		std::cout << " done" << std::endl;
		file.close();
	}

	std::ifstream* NeuralNetwork::deserialise(std::ifstream* _stream, bool _swapEndianness) {
		readVariable(&m_learningRate, _stream, _swapEndianness);
		readVariable(&m_activationFunction, _stream, _swapEndianness);
		//readVariable(&m_hardwareMode, _stream, _swapEndianness);

		m_networkStructure.clear();
		readVector(&m_networkStructure, _stream, _swapEndianness);
		m_biasMatrixes.clear();
		readVector(&m_biasMatrixes, _stream, _swapEndianness);
		for (int i = 0; i < m_weightMatrixes.size(); i++) {
			if (m_hardwareMode == CUDA) m_weightMatrixes[i]->freeMemoryGPU();
			if (m_hardwareMode == CPU) m_weightMatrixes[i]->freeMemoryCPU();
		}
		m_weightMatrixes.clear();
		readMatrixVector(&m_weightMatrixes, _stream, _swapEndianness);

		m_networkErrors.clear();

		for (int i = 0; i < m_networkLength; i++) {
			std::vector <double> errorMatrix;
			for (int j = 0; j < m_networkStructure.at(i); j++) {
				errorMatrix.push_back(0);
			}
			m_networkErrors.push_back(errorMatrix);
		}

		initNetwork();

		return _stream;
	}
	void NeuralNetwork::deserialise(const char* _path, bool _swapEndianness) {
		std::cout << "[CONVEX] Loading network from " << _path << "...";

		std::ifstream file;
		file.open(_path, std::ios::binary);

		if (file.is_open()) {
			deserialise(&file, _swapEndianness);

			std::cout << " done" << std::endl;
			file.close();
		}
		else {
			std::cout << " failed" << std::endl;
			std::cerr << "[CONVEX] ERROR: Could not open " << _path << std::endl;
		}
	}

	double NeuralNetwork::normalise(double _input) {
		switch (m_activationFunction) {
		case SIGMOID:
			return 1 / (1 + exp(-_input));
			break;
		case TANH:
			return 2 / (1 + exp(-2 * _input));
			break;
		case RELU:
			return (_input < 0) ? 0 : _input;
			break;
		case NONE:
			return _input;
			break;
		default:
			std::cerr << "[CONVEX ERROR] Invalid activation function" << std::endl;
			return 0;
			break;
		}
	}

	double NeuralNetwork::deriveNormalise(double _input) {
		switch (m_activationFunction) {
		case SIGMOID:
			return (1 / (1 + exp(-_input)))*(1 - (1 / (1 + exp(-_input))));
			//return _input * (1 - _input);
			break;
		case TANH:
			return 1 - std::pow((2 / (1 + exp(-2 * _input))), 2);
			break;
		case RELU:
			return _input;
			break;
		case NONE:
			return _input;
			break;
		default:
			std::cerr << "[CONVEX ERROR] Invalid activation function" << std::endl;
			return 0;
			break;
		}
	}

	double* NeuralNetwork::getWeight(int _n1Layer, int _n1Neuron, int _n2Layer, int _n2Neuron) {
		return &m_weightMatrixes.at(_n2Layer)->at2D(_n2Neuron, _n1Neuron);
	}

	void NeuralNetwork::matrixMultiplyCPU(Matrix <double>* _matrixA, Matrix <double>* _matrixB, Matrix <double>* output) {
		unsigned int rows = (unsigned int)_matrixA->dimensions[0];
		unsigned int cols = (unsigned int)_matrixB->dimensions[1];

		#pragma omp parallel for
		{
			for (unsigned int i = 0; i < rows; i++) {
				for (unsigned int j = 0; j < cols; j++) {
					for (unsigned int k = 0; k < _matrixB->dimensions[0]; k++) {
						output->at2D(i, j) += _matrixA->at2D(i, k) * _matrixB->at2D(j, k);
					}
				}
			}
		}
	}
}