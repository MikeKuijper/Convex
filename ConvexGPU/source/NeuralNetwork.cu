#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "Convex.h"
#include "ConvexGPU.h"

//using namespace Convex;

namespace ConvexGPU {
	NeuralNetwork::NeuralNetwork() {}

	NeuralNetwork::NeuralNetwork(std::vector <int> _structure) {
		m_networkStructure = _structure;
		m_networkLength = _structure.size();
		m_learningRate = 0.1;
		m_activationFunction = SIGMOID;
		//m_hardwareMode = CPU;

		generateNetwork();
	}

	NeuralNetwork::NeuralNetwork(const char* _path) {
		//deserialise(_path);
	}

	void NeuralNetwork::generateNetwork() {
		m_weightMatrixes.clear();
		m_biasMatrixes.clear();
		m_networkErrors.clear();

		for (int i = 1; i < m_networkLength; i++) {
			//double* weightMatrix = (double*) malloc(sizeof(double) * m_networkStructure.at(i - 1) * m_networkStructure.at(i));
			Matrix<double> weightMatrix(m_networkStructure.at(i - 1), m_networkStructure.at(i));

			for (int i = 0; i < weightMatrix.dimensions[0]; i++) {
				for (int j = 0; j < weightMatrix.dimensions[1]; j++) {
					weightMatrix.at2D(i, j) = randomNumber(-2.0f, 2.0f);
				}
			}
			weightMatrix.copyMemoryToDevice();
			m_weightMatrixes.push_back(weightMatrix);
		}

		for (int i = 0; i < m_networkLength; i++) {
			std::vector <double> biasMatrix;
			std::vector <double> errorMatrix;
			for (int j = 0; j < m_networkStructure.at(i); j++) {
				biasMatrix.push_back(randomNumber(-2, 2));
				errorMatrix.push_back(0);
			}
			m_biasMatrixes.push_back(biasMatrix);
			m_networkErrors.push_back(errorMatrix);
		}
	}

	Matrix<double> NeuralNetwork::feed(Matrix<double>* _input) {
		return feed(_input, 0, m_weightMatrixes.size());
	}

	Matrix<double> NeuralNetwork::feed(Matrix<double>* _input, int _rangeStart, int _rangeEnd) {
		//int activationID = 0;
		m_activations.clear();

		m_activations.push_back(*_input);
		for (int i = _rangeStart; i < _rangeEnd; i++) {
			//std::cout << "ACTIVATION:" << std::endl;
			//m_activations[m_activations.size() - 1].print();
			//std::cout << "WEIGHTS:" << std::endl;
			//m_weightMatrixes.at(i).print();
			//std::cout << m_weightMatrixes.at(i) << std::endl;

			timeExecution([this, i]() {
				m_activations[m_activations.size() - 1].copyMemoryToDevice();
				m_weightMatrixes.at(i).copyMemoryToDevice();
				//m_weightMatrixes.at(0).print();
			}, "=== Memory copy", 2);
			
			Matrix <double> result;
			timeExecution([this, i, &result]() {
				result = matrixMultiply2D(&m_activations[m_activations.size() - 1], &m_weightMatrixes.at(i));
			}, "=== Matrix Multiplication", 2);

			//TODO Write normalisation kernel and implement biases in previous layer
			for (int j = 0; j < result.dimensions.at(1); j++) {
				result.at2D(0, j) = normalise(result.at2D(0, j) + m_biasMatrixes.at(i + 1).at(j));
			}
			result.copyMemoryToDevice();

			//std::cout << "RESULT:" << std::endl;
			//result.print();

			//activationID++;
			m_activations.push_back(result);
		}

		//m_activations[m_activations.size() - 1].flatten();
		return m_activations[m_activations.size() - 1];
	}

	Matrix<double> NeuralNetwork::train(Matrix<double>* _input, Matrix<double>* _targetOutput) {
		return train(_input, _targetOutput, 0, m_networkLength - 1);
	}

	//TODO Check if this still works (cost reduces)
	Matrix<double> NeuralNetwork::train(Matrix<double>* _input, Matrix<double>* _targetOutput, int _rangeStart, int _rangeEnd) {
		Matrix<double> feedResult;

		timeExecution([&feedResult, this, _input, _rangeStart, _rangeEnd]() {
			feedResult = feed(_input, _rangeStart, _rangeEnd);
		}, "=== Feed", 2);

		timeExecution([this, &feedResult, _targetOutput, _rangeEnd, _rangeStart]() {
			m_cost = 0;
			m_globalError = 0;

			for (int i = 0; i < feedResult.size(); i++) {
				double gradient = -deriveNormalise(feedResult.at2D(0, i));
				double error = feedResult.at2D(0, i) - _targetOutput->at2D(0, i);

				m_networkErrors.at(_rangeEnd).at(i) = error * gradient;
				m_cost += error;
			}

			if (m_learningRate != 0) {
				timeExecution([this, _rangeEnd, _rangeStart]() {
					for (int i = _rangeEnd - 1; i >= _rangeStart; i--) {
						std::vector <double> nextLayerErrors = m_networkErrors.at(i + 1);

						for (int j = 0; j < m_networkStructure.at(i); j++) {
							double sum = 0;

							timeExecution([this, &sum, nextLayerErrors, i, j]() {
								for (int k = 0; k < nextLayerErrors.size(); k++) {
									//double* weight = getWeight(i + 1, k, i, j);
									double* weight = &m_weightMatrixes.at(i).at2D(j, k);								
									*weight += m_learningRate * nextLayerErrors.at(k) * m_activations.at(i).at2D(0, j);
						
									sum += *weight * nextLayerErrors.at(k);
								}
							}, "======= Train-third", 3);

							double currentError = sum * deriveNormalise(m_activations.at(i).at2D(0, j));
							m_networkErrors.at(i).at(j) = currentError;
							m_globalError += abs(currentError);
							m_biasMatrixes.at(i).at(j) += m_learningRate * currentError;
						}
					}
				}, "===== Train-second", 2);
			}
		}, "=== Train sequence", 1);

		return feedResult;
	}

	double NeuralNetwork::train(ImageClassDataset* _dataset) {
		//double cost = 0;
		//double score = 0;
		//for (int n = 0; n < _dataset->m_size; n++) {
		//	std::vector <double> targetResult(10, 0);
		//	targetResult[_dataset->m_labels[n]] = 1;
		//	auto feedResult = train(&_dataset->m_imagesFlattened[n], &targetResult);
		//	cost += m_cost / _dataset->m_size;
//
//			std::vector<double>::iterator maxElement = std::max_element(feedResult.begin(), feedResult.end());
//			int maxElementIndex = std::distance(feedResult.begin(), maxElement);
//
//			//std::cout << _dataset->m_labels[n] << ", " << maxElementIndex << std::endl;
//			if (maxElementIndex == _dataset->m_labels[n]) {
//				score++;
//			}
//
//			//if (n % 1000 == 0) std::cout << "[CONVEX] Subtrain cycle " << n << ": \t" << m_cost << std::endl;
//		}
//		this->m_score = score / _dataset->m_size;
//		return cost;
		return 0;
	}

	void NeuralNetwork::trainSequence(ImageClassDataset* _dataset, int _epochs, const char* _path) {
		std::cout << "[CONVEX] Training start" << std::endl;

		double minCost = assess(_dataset);
		int nonImprovementCount = 0;

		for (int n = 0; n < _epochs; n++) {
			double cost = train(_dataset);
			std::cout << "[CONVEX] Train epoch " << n << ": " << cost << " (" << round((float)m_score * 10000) / 100 << "%)" << std::endl;

			if (cost <= minCost) {
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

	double NeuralNetwork::assess(ImageClassDataset* _dataset) {
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

	//TODO add array dimensions object at network generation to solve errors at line 275 and 276 (rowsB and colsB)
	// OR custom matrix/array class
	//std::vector<std::vector <double>> NeuralNetwork::matrixMultiply(double* _matrixA, double* _matrixB) {
		//int rowsA = _matrixA->size();
		//int colsA = _matrixA->at(0).size();
		//int rowsB = _matrixB->size();
		//int colsB = _matrixB->at(0).size();
		//int rowsC = rowsA;
		//int colsC = colsB;

		//std::vector <double> aV = flattenVector(*_matrixA);
		//std::vector <double> bV = flattenVector(*_matrixB);
		//std::vector<std::vector <double>> cV(rowsC, std::vector <double>(colsC, 0));

		//double* a = &aV[0];
		//double* b = &bV[0];
		//double* c = &cV[0][0];

		//double* deviceA;
		//double* deviceB;
		//double* deviceC;

		//cudaMalloc((void**) &deviceA, sizeof(double) * rowsA * colsA);
		//cudaMalloc((void**) &deviceB, sizeof(double) * rowsB * colsB);
		//cudaMalloc((void**) &deviceC, sizeof(double) * rowsC * colsC);

		//cudaMemcpy(deviceA, &b, bV.size() * sizeof(double), cudaMemcpyHostToDevice);
		//cudaMemcpy(deviceB, &b, bV.size() * sizeof(double), cudaMemcpyHostToDevice);
		//cudaMemcpy(deviceC, &c, cV.size() * cV.at(0).size() * sizeof(double), cudaMemcpyHostToDevice);

		//matrixMultiplication(deviceA, deviceB, deviceC, rowsC, colsC);

		//cudaMemcpy(_matrixA, deviceA, _matrixA->size() * sizeof(double), cudaMemcpyDeviceToHost);
		//cudaMemcpy(_matrixB, deviceB, _matrixB->size() * sizeof(double), cudaMemcpyDeviceToHost);
		//cudaMemcpy(&c, deviceC, cV.size() * cV.at(0).size() * sizeof(double), cudaMemcpyDeviceToHost);

		//std::vector <double> subresult(c, c + (sizeof (c) / sizeof (double)));
		//std::vector <std::vector<double>> result(1, subresult);
		//return result;
	//}

	void NeuralNetwork::freeMemory() {
		for (int i = 0; i < m_activations.size() - 2; i++) {
			m_activations[i].freeMemory();
		}
	}

	/*std::ofstream* NeuralNetwork::serialise(std::ofstream* _outputStream, bool _swapEndianness) {
		char signature[10] = "#ConvexNN";
		writeVariable(&signature, _outputStream);

		writeVariable(&m_learningRate, _outputStream);
		writeVariable(&m_activationFunction, _outputStream);
		writeVariable(&m_hardwareMode, _outputStream);

		writeVector(&m_networkStructure, _outputStream);
		writeVector(&m_biasMatrixes, _outputStream);
		writeVector(&m_weightMatrixes, _outputStream);

		return _outputStream;
	}*/

	/*void NeuralNetwork::serialise(const char * _path, bool _swapEndianness) {
		std::cout << "[CONVEX] Saving network to " << _path << "...";

		std::ofstream file;
		file.open(_path, std::ios::binary);

		serialise(&file, _swapEndianness);

		std::cout << " done" << std::endl;
		file.close();
	}*/

	/*std::ifstream* NeuralNetwork::deserialise(std::ifstream* _inputStream, bool _swapEndianness) {
		_inputStream->seekg(10, std::ios::cur);

		readVariable(&m_learningRate, _inputStream);
		readVariable(&m_activationFunction, _inputStream);
		readVariable(&m_hardwareMode, _inputStream);

		readVector(&m_networkStructure, _inputStream);
		readVector(&m_biasMatrixes, _inputStream);
		readVector(&m_weightMatrixes, _inputStream);

		return _inputStream;
	}*/

	/*void NeuralNetwork::deserialise(const char* _path, bool _swapEndianness) {
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
			std::cout << "[CONVEX] ERROR: Could not open " << _path << std::endl;
		}
	}*/

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
			std::cout << "[CONVEX ERROR] Invalid activation function" << std::endl;
			return 0;
			break;
		}
	}

	double NeuralNetwork::deriveNormalise(double _input) {
		switch (m_activationFunction) {
		case SIGMOID:
			return (1 / (1 + exp(-_input)))*(1-(1 / (1 + exp(-_input))));
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
			std::cout << "[CONVEX ERROR] Invalid activation function" << std::endl;
			return 0;
			break;
		}
	}

	double* NeuralNetwork::getWeight(int _n1Layer, int _n1Neuron, int _n2Layer, int _n2Neuron) {
		return &m_weightMatrixes.at(_n2Layer).at2D(_n2Neuron, _n1Neuron);
	}
}