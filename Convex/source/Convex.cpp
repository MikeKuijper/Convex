#include "Convex.h"

namespace Convex {
	NeuralNetwork::NeuralNetwork() {}

	NeuralNetwork::NeuralNetwork(std::vector <int> _structure) {
		m_networkStructure = _structure;
		m_networkLength = (unsigned int) _structure.size();
		m_learningRate = 0.1;
		m_activationFunction = SIGMOID;
		//m_hardwareMode = CPU;

		generateNetwork();
	}

	NeuralNetwork::NeuralNetwork(const char* _path) {
		deserialise(_path);
	}

	void NeuralNetwork::generateNetwork() {
		m_weightMatrixes.clear();
		m_biasMatrixes.clear();
		m_networkErrors.clear();

		for (int i = 0; i < m_networkLength; i++) {
			if (i != 0) {
				std::vector <std::vector <double>> matrix;
				for (int j = 0; j < m_networkStructure.at(i - 1); j++) {
					std::vector <double> submatrix;
					for (int k = 0; k < m_networkStructure.at(i); k++) {
						submatrix.push_back(randomNumber(-2, 2));
					}
					matrix.push_back(submatrix);
				}
				m_weightMatrixes.push_back(matrix);
			}

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

	std::vector <double> NeuralNetwork::feed(std::vector <double>* _input) {
		std::vector <std::vector <double>> activation;
		activation.push_back(*_input);

		m_activations.clear();

		m_activations.push_back(*_input);
		for (unsigned int i = 0; i < m_weightMatrixes.size(); i++) {
			activation = matrixMultiply(&activation, &m_weightMatrixes.at(i));
			for (unsigned int j = 0; j < activation.at(0).size(); j++) {
				activation.at(0).at(j) = normalise(activation.at(0).at(j) + m_biasMatrixes.at(i + 1).at(j));
			}
			m_activations.push_back(activation.at(0));
		}
		return activation.at(0);
	}

	std::vector <double> NeuralNetwork::feed(std::vector <double>* _input, unsigned int _rangeStart, unsigned int _rangeEnd) {
		std::vector <std::vector <double>> activation;
		activation.push_back(*_input);

		m_activations.clear();

		m_activations.push_back(*_input);
		for (unsigned int i = _rangeStart; i < _rangeEnd; i++) {
			activation = matrixMultiply(&activation, &m_weightMatrixes.at(i));
			for (unsigned int j = 0; j < activation.at(0).size(); j++) {
				activation.at(0).at(j) = normalise(activation.at(0).at(j) + m_biasMatrixes.at(i + 1).at(j));
			}
			m_activations.push_back(activation.at(0));
		}
		return activation.at(0);
	}

	std::vector <double> NeuralNetwork::train(std::vector <double>* _input, std::vector <double>* _targetOutput) {
		return train(_input, _targetOutput, 0, m_networkLength - 1);
	}

	//TODO Check if this still works (cost reduces)
	std::vector <double> NeuralNetwork::train(std::vector <double>* _input, std::vector <double>* _targetOutput, int _rangeStart, int _rangeEnd) {
		std::vector <double> feedResult;

		timeExecution([&feedResult, this, _input, _rangeStart, _rangeEnd]() {
			feedResult = feed(_input, _rangeStart, _rangeEnd);
		}, "=== Feed", 2);

		timeExecution([this, &feedResult, _targetOutput, _rangeEnd, _rangeStart]() {
			m_cost = 0;
			m_globalError = 0;

			for (int i = 0; i < feedResult.size(); i++) {
				double gradient = -deriveNormalise(feedResult.at(i));
				double error = feedResult.at(i) - _targetOutput->at(i);
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
									double* weight = getWeight(i + 1, k, i, j);
									*weight += m_learningRate * nextLayerErrors.at(k) * m_activations.at(i).at(j);

									sum += *weight * nextLayerErrors.at(k);
								}
								}, "======= Weight loop", 3);

							double currentError = sum * deriveNormalise(m_activations.at(i).at(j));
							m_networkErrors.at(i).at(j) = currentError;
							m_globalError += abs(currentError);
							m_biasMatrixes.at(i).at(j) += m_learningRate * currentError;
						}
					}
				}, "===== Layer loop", 2);
			}
		}, "=== Train sequence", 1);

		return feedResult;
	}

	double NeuralNetwork::train(Convex::ImageClassDataset* _dataset, bool _assessPerformance) {
		if (_assessPerformance == true) {
			double cost = 0;
			double score = 0;
			for (unsigned int n = 0; n < _dataset->m_size; n++) {
				std::vector <double> targetResult(10, 0);
				targetResult[_dataset->m_labels[n]] = 1;

				auto feedResult = train(&_dataset->m_imagesFlattened[n], &targetResult);
				cost += m_cost / _dataset->m_size;

				std::vector<double>::iterator maxElement = std::max_element(feedResult.begin(), feedResult.end());
				int maxElementIndex = std::distance(feedResult.begin(), maxElement);

				if (maxElementIndex == _dataset->m_labels[n]) {
					score++;
				}
			}
			this->m_score = score / _dataset->m_size;
			return cost;
		}
		else {
			double cost = 0;
			for (unsigned int n = 0; n < _dataset->m_size; n++) {
				std::vector <double> targetResult(10, 0);
				targetResult[_dataset->m_labels[n]] = 1;

				auto feedResult = train(&_dataset->m_imagesFlattened[n], &targetResult);
				cost += m_cost / _dataset->m_size;
			}
			return cost;
		}
	}

	void NeuralNetwork::trainSequence(Convex::ImageClassDataset* _dataset, int _epochs, const char* _path) {
		std::cout << "[CONVEX] Training start" << std::endl;

		double minCost = abs(assess(_dataset));
		int nonImprovementCount = 0;

		for (int n = 0; n < _epochs; n++) {
			double cost = train(_dataset, n % 1 == 0);
			if (n % 1 == 0) std::cout << "[CONVEX] Train epoch " << n << ": " << cost << " (" << round((float)m_score * 10000) / 100 << "%)" << std::endl;
			else std::cout << "[CONVEX] Train epoch " << n << ": " << cost << std::endl;

			if (abs(cost) <= minCost) {
				serialise(_path);

				//_dataset->shuffle();

				minCost = abs(cost);
				nonImprovementCount = 0;
			}
			else {
				nonImprovementCount++;
				if (nonImprovementCount == 64) {
					double newLearningRate = m_learningRate * pow(0.5, 0.1f);
					deserialise(_path);
					m_learningRate = newLearningRate;

					nonImprovementCount = 0;
					std::cout << "[CONVEX] Adjusting learning rate to " << m_learningRate << std::endl;
				}
			}
		}
	}

	double NeuralNetwork::assess(Convex::ImageClassDataset* _dataset) {
		std::cout << "[CONVEX] Assessing network...";

		double learningRateTemp = m_learningRate;
		m_learningRate = 0;

		double cost = train(_dataset, true);
		m_learningRate = learningRateTemp;

		std::cout << " done" << std::endl;
		std::cout << "[CONVEX] Cost: " << cost << " (" << round((float)m_score * 10000) / 100 << "%)" << std::endl;
		return cost;
	}

	/*std::vector <double> NeuralNetwork::train(std::vector <double>* _input, std::vector <double>* _targetOutput, std::vector <std::vector <std::vector <double>>>* _ptrWeightMatrixes, std::vector<std::vector <double>>* _ptrBiasMatrixes) {
		std::vector <double> feedResult = feed(_input);

		m_cost = 0;
		m_globalError = 0;

		for (unsigned int i = 0; i < feedResult.size(); i++) {
			double gradient = deriveNormalise(feedResult.at(i));
			double error = _targetOutput->at(i) - feedResult.at(i);
			m_networkErrors.at(m_networkLength - 1).at(i) = error * gradient;
			m_cost += abs(error);
		}

		for (int i = m_networkLength - 2; i >= 0; i--) {
			std::vector <double> nextLayerErrors = m_networkErrors.at(i + 1);
			for (int j = 0; j < m_networkStructure.at(i); j++) {
				double sum = 0;
				for (unsigned int k = 0; k < nextLayerErrors.size(); k++) {
					double* weight = getWeight(i + 1, k, i, j, _ptrWeightMatrixes);

					*weight += m_learningRate * nextLayerErrors.at(k) * m_activations.at(i).at(j);
					sum += *weight * nextLayerErrors.at(k);
				}

				double currentError = sum * deriveNormalise(m_activations.at(i).at(j));
				m_networkErrors.at(i).at(j) = currentError;
				m_globalError += abs(currentError);

				_ptrBiasMatrixes->at(i).at(j) += m_learningRate * currentError;
			}
		}

		return feedResult;
	}*/

	std::vector<std::vector <double>> NeuralNetwork::matrixMultiply(std::vector<std::vector <double>>* _matrixA, std::vector<std::vector <double>>* _matrixB) {
		unsigned int rows = (unsigned int) _matrixA->size();
		unsigned int cols = (unsigned int) _matrixB->at(0).size();

		std::vector<std::vector <double>> output(rows, std::vector <double>(cols, 0));

		#pragma omp parallel for
		{
			for (unsigned int i = 0; i < rows; i++) {
				for (unsigned int j = 0; j < cols; j++) {
					for (unsigned int k = 0; k < _matrixB->size(); k++) {
						output[i][j] += _matrixA->at(i).at(k) * _matrixB->at(k).at(j);
					}
				}
			}
		}
		return output;
	}

	std::ofstream* NeuralNetwork::serialise(std::ofstream* _outputStream, bool _swapEndianness) {
		//char signature[10] = "#ConvexNN";
		//writeVariable(&signature, _outputStream, _swapEndianness);

		writeVariable(&m_learningRate, _outputStream, _swapEndianness);
		writeVariable(&m_activationFunction, _outputStream, _swapEndianness);
		//writeVariable(&m_hardwareMode, _outputStream, _swapEndianness);

		writeVector(&m_networkStructure, _outputStream, _swapEndianness);
		writeVector(&m_biasMatrixes, _outputStream, _swapEndianness);
		writeVector(&m_weightMatrixes, _outputStream, _swapEndianness);

		return _outputStream;
	}

	void NeuralNetwork::serialise(const char * _path, bool _swapEndianness) {
		std::cout << "[CONVEX] Saving network to " << _path << "...";

		std::ofstream file;
		file.open(_path, std::ios::binary);

		serialise(&file, _swapEndianness);

		std::cout << " done" << std::endl;
		file.close();
	}

	std::ifstream* NeuralNetwork::deserialise(std::ifstream* _inputStream, bool _swapEndianness) {
		//_inputStream->seekg(10, std::ios::cur);

		readVariable(&m_learningRate, _inputStream, _swapEndianness);
		readVariable(&m_activationFunction, _inputStream, _swapEndianness);
		//readVariable(&m_hardwareMode, _inputStream, _swapEndianness);

		m_networkStructure.clear();
		readVector(&m_networkStructure, _inputStream, _swapEndianness);
		m_networkLength = (unsigned int) m_networkStructure.size();
		m_biasMatrixes.clear();
		readVector(&m_biasMatrixes, _inputStream, _swapEndianness);
		m_weightMatrixes.clear();
		readVector(&m_weightMatrixes, _inputStream, _swapEndianness);

		m_networkErrors.clear();

		for (int i = 0; i < m_networkLength; i++) {
			std::vector <double> errorMatrix;
			for (int j = 0; j < m_networkStructure.at(i); j++) {
				errorMatrix.push_back(0);
			}
			m_networkErrors.push_back(errorMatrix);
		}

		return _inputStream;
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
			std::cerr << "[CONVEX] ERROR: Invalid activation function" << std::endl;
			break;
		}
	}

	double NeuralNetwork::deriveNormalise(double _input) {
		switch (m_activationFunction) {
		case SIGMOID:
			//return _input * (1 - _input);
			return (1 / (1 + exp(-_input)))*(1 - (1 / (1 + exp(-_input))));
			break;
		case TANH:
			return 1 - std::pow((2 / (1 + exp(-2 * _input))), 2);
			break;
		case RELU:
			return 1;
			break;
		case NONE:
			return _input;
			break;
		default:
			std::cerr << "[CONVEX] ERROR: Invalid activation function" << std::endl;
			break;
		}
	}

	double* NeuralNetwork::getWeight(int _n1Layer, int _n1Neuron, int _n2Layer, int _n2Neuron) {
		return &m_weightMatrixes.at(_n2Layer).at(_n2Neuron).at(_n1Neuron);
	}

	double* NeuralNetwork::getWeight(int _n1Layer, int _n1Neuron, int _n2Layer, int _n2Neuron, std::vector<std::vector<std::vector<double>>>* _ptrWeightMatrixes) {
		return &_ptrWeightMatrixes->at(_n2Layer).at(_n2Neuron).at(_n1Neuron);
	}

	AutoEncoder::AutoEncoder() {}

	AutoEncoder::AutoEncoder(std::vector <int> _structure) {
		m_fullNetworkStructure = _structure;
		m_sharedLayer = std::floor(m_fullNetworkStructure.size() / 2);

		m_learningRate = 1;
		m_activationFunction = SIGMOID;

		generateNetworks();
	}

	AutoEncoder::AutoEncoder(std::vector <int> _structure, int _sharedLayer) {
		m_fullNetworkStructure = _structure;
		m_sharedLayer = _sharedLayer;

		m_learningRate = 1;
		m_activationFunction = SIGMOID;

		generateNetworks();
	}

	void AutoEncoder::generateNetworks() {
		m_encoderNetworkStructure = subVector(&m_fullNetworkStructure, 0, m_sharedLayer);
		m_decoderNetworkStructure = subVector(&m_fullNetworkStructure, (unsigned int) m_sharedLayer, (unsigned int) m_fullNetworkStructure.size() - 1);

		m_network = NeuralNetwork(m_fullNetworkStructure);
		m_network.m_activationFunction = this->m_activationFunction;
		m_network.m_learningRate = this->m_learningRate;
	}

	std::vector <double> AutoEncoder::encode(std::vector <double>* _input) {
		return m_network.feed(_input, 0, m_sharedLayer);
	}

	std::vector <double> AutoEncoder::decode(std::vector <double>* _input) {
		return m_network.feed(_input, m_sharedLayer, m_fullNetworkStructure.size() - 1);
	}

	std::vector <double> AutoEncoder::feed(std::vector <double>* _input) {
		return m_network.feed(_input);
	}

	NeuralNetwork AutoEncoder::getEncoderNetwork() {
		NeuralNetwork encoder;
		encoder.m_weightMatrixes = subVector(&m_network.m_weightMatrixes, 0, m_sharedLayer - 1);
		encoder.m_biasMatrixes = subVector(&m_network.m_biasMatrixes, 0, m_sharedLayer);
		encoder.m_networkErrors = subVector(&m_network.m_networkErrors, 0, m_sharedLayer);
		encoder.m_activationFunction = m_activationFunction;
		return encoder;
	}

	NeuralNetwork AutoEncoder::getDecoderNetwork() {
		NeuralNetwork decoder;
		decoder.m_weightMatrixes = subVector(&m_network.m_weightMatrixes, m_sharedLayer, m_network.m_weightMatrixes.size() - 1);
		decoder.m_biasMatrixes = subVector(&m_network.m_biasMatrixes, m_sharedLayer - 1, m_network.m_biasMatrixes.size() - 1);
		decoder.m_networkErrors = subVector(&m_network.m_networkErrors, m_sharedLayer - 1, m_network.m_networkErrors.size() - 1);
		decoder.m_activationFunction = m_activationFunction;
		return decoder;
	}

	double AutoEncoder::train(std::vector <double>* _input) {
		m_network.m_learningRate = m_learningRate;
		m_network.m_activationFunction = m_activationFunction;

		m_network.train(_input, _input);
		return m_network.m_cost;
	}

	std::ofstream* AutoEncoder::serialise(std::ofstream* _outputStream, bool _swapEndianness) {
		char signature[10] = "#ConvexAE";
		writeVariable(&signature, _outputStream, _swapEndianness);

		writeVariable(&m_sharedLayer, _outputStream, _swapEndianness);
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
		_inputStream->seekg(10, std::ios::cur);

		readVariable(&m_sharedLayer, _inputStream, _swapEndianness);
		readVariable(&m_learningRate, _inputStream, _swapEndianness);
		readVariable(&m_activationFunction, _inputStream, _swapEndianness);
		readVector(&m_fullNetworkStructure, _inputStream, _swapEndianness);

		m_network.deserialise(_inputStream, _swapEndianness);

		m_encoderNetworkStructure = subVector(&m_fullNetworkStructure, 0, m_sharedLayer);
		m_decoderNetworkStructure = subVector(&m_fullNetworkStructure, m_sharedLayer, m_fullNetworkStructure.size() - 1);

		return _inputStream;
	}

	void AutoEncoder::deserialise(const char* _path) {
		std::ifstream file;
		file.open(_path, std::ios::binary);

		deserialise(&file);
		file.close();
	}

	ImageClassDataset::ImageClassDataset() {
		return;
	}

	ImageClassDataset::ImageClassDataset(const char* _imagePath, const char* _labelPath, bool _swapEndianness) {
		deserialiseMNIST(_imagePath, _labelPath, _swapEndianness);
	}

	std::ifstream* ImageClassDataset::deserialiseMNISTImages(std::ifstream* _inputStream, bool _swapEndianness) {
		_inputStream->seekg(4, std::ios::cur);
		readVariable(&m_size, _inputStream, true);
		readVariable(&m_rows, _inputStream, true);
		readVariable(&m_columns, _inputStream, true);
		std::cout << "[CONVEX] Loading MNIST image file...";
		//std::cout << "[CONVEX] Size: \t\t" << m_size << std::endl;
		//std::cout << "[CONVEX] Rows: \t\t" << m_rows << std::endl;
		//std::cout << "[CONVEX] Columns: \t" << m_columns << std::endl;

		for (unsigned int n = 1; n <= m_size; n++) {
			std::vector <std::vector<double>> image;
			for (unsigned int y = 0; y < m_rows; y++) {
				std::vector <double> row;
				for (unsigned int x = 0; x < m_columns; x++) {
					unsigned char value;
					readVariable(&value, _inputStream, true);
					row.push_back((double)value);
				}
				image.push_back(row);
			}
			m_images.push_back(image);
			//if (n % 1000 == 0) std::cout << "[CONVEX] " << n << "\t (" << round((float)n / m_size * 1000) / 10 << "%)" << std::endl;
		}

		std::cout << " done" << std::endl;

		return _inputStream;
	}

	void ImageClassDataset::deserialiseMNISTImages(const char* _path, bool _swapEndianness) {
		std::ifstream imageFileStream;
		imageFileStream.open(_path, std::ios::binary);

		if (imageFileStream.is_open()) {
			deserialiseMNISTImages(&imageFileStream);
			imageFileStream.close();
		}
		else {
			std::cout << " failed" << std::endl;
			std::cout << "[CONVEX] ERROR: Could not open " << _path << std::endl;
		}
	}

	std::ifstream* ImageClassDataset::deserialiseMNISTLabels(std::ifstream* _inputStream, bool _swapEndianness) {
		_inputStream->seekg(4, std::ios::cur);
		readVariable(&m_size, _inputStream, _swapEndianness);

		std::cout << "[CONVEX] Loading MNIST label file...";

		for (unsigned int n = 0; n < m_size; n++) {
			unsigned char label;
			readVariable(&label, _inputStream, _swapEndianness);
			m_labels.push_back(label);
		}

		std::cout << " done" << std::endl;

		return _inputStream;
	}

	void ImageClassDataset::deserialiseMNISTLabels(const char* _path, bool _swapEndianness) {
		std::ifstream labelFileStream;
		labelFileStream.open(_path, std::ios::binary);

		if (labelFileStream.is_open()) {
			deserialiseMNISTImages(&labelFileStream);
			labelFileStream.close();
		}
		else {
			std::cout << " failed" << std::endl;
			std::cerr << "[CONVEX] ERROR: Could not open " << _path << std::endl;
		}
	}

	void ImageClassDataset::deserialiseMNIST(const char* _imagePath, const char* _labelPath, bool _swapEndianness) {
		std::ifstream imageFileStream;
		imageFileStream.open(_imagePath, std::ios::binary);

		if (imageFileStream.is_open()) {
			deserialiseMNISTImages(&imageFileStream, _swapEndianness);
			imageFileStream.close();
		}
		else {
			std::cout << " failed" << std::endl;
			std::cerr << "[CONVEX] ERROR: Could not open " << _imagePath << std::endl;
		}

		std::ifstream labelFileStream;
		labelFileStream.open(_labelPath, std::ios::binary);

		if (labelFileStream.is_open()) {
			deserialiseMNISTLabels(&labelFileStream, _swapEndianness);
			labelFileStream.close();
		}
		else {
			std::cout << " failed" << std::endl;
			std::cerr << "[CONVEX] ERROR: Could not open " << _labelPath << std::endl;
		}
	}

	void ImageClassDataset::flatten() {
		for (unsigned int i = 0; i < m_size; i++) {
			m_imagesFlattened.push_back(flattenVector(m_images.at(i)));
		}
		m_images.clear();
	}

	void ImageClassDataset::flattenMatrix() {
		for (unsigned int i = 0; i < m_size; i++) {
			//m_imagesFlattenedMatrix.push_back(flattenVector(m_images.at(i)));
		}
		m_images.clear();
	}

	void ImageClassDataset::shuffle() {
		unsigned seed = (unsigned) std::chrono::system_clock::now()
			.time_since_epoch()
			.count();
		std::shuffle(std::begin(m_imagesFlattened), std::end(m_imagesFlattened), std::default_random_engine(seed));
	}
}