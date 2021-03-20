#pragma once
#include "Convex.h"
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
	double randomNumber(double _min, double _max) {
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_real_distribution<> dis(_min, _max);
		//return dis(gen);
		return 0.2;
	}

	void testSetupIncrement();
	void testSetupMatrixMultiply();

	template<class T>
	class Matrix {
	public:
		std::vector <int> dimensions;
		std::vector <int> dimensionsReversed;
		T* m_hostData;
		T* m_deviceData;
		int m_size;

		Matrix() {
			return;
		}

		~Matrix() {
			//cudaFree(m_deviceData);
			//freeMemory();
		}

		Matrix(std::vector<int> dim) {
			dimensions = dim;
			dimensionsReversed = dim;

			std::reverse(dimensionsReversed.begin(), dimensionsReversed.end());

			int size = 1;
			for (int i = 0; i < dimensionsReversed.size(); i++) {
				size *= dimensionsReversed.at(i);
			}

			m_hostData = (T*)malloc(sizeof(T) * size);
			cudaMallocManaged((void**)&m_deviceData, sizeof(T) * size);

			cudaDeviceSynchronize();
			m_size = size;

			//std::cout << "[CONVEX] Allocated " << size * sizeof(T) << " bytes of GPU memory for " << size << " elements" << std::endl;
		}

		//TODO: allow for use with standard memory and AMD GPUs
		template <typename... Args>
		Matrix(Args... dims) {
			dimensionsReversed = { static_cast<int>(dims)... };
			dimensions = { static_cast<int>(dims)... };

			std::reverse(dimensionsReversed.begin(), dimensionsReversed.end());

			int size = 1;
			for (int i = 0; i < dimensions.size(); i++) {
				size *= dimensions.at(i);
			}

			m_hostData = (T*)malloc(sizeof(T) * size);
			cudaMallocManaged((void**)&m_deviceData, sizeof(T) * size);

			cudaDeviceSynchronize();
			m_size = size;

			//std::cout << "[CONVEX] Allocated " << size * sizeof(T) << " bytes of GPU memory for " << size << " elements" << std::endl;
		}

		void flatten() {
			dimensions = { m_size };
			dimensionsReversed = { m_size };
		}

		void addDimension() {
			dimensions.insert(dimensionsReversed.begin(), 1);
			dimensionsReversed.push_back(1);
		}

		template <typename... Args>
		int getIndex(Args... indexes) {
			std::vector <int> dim = { indexes... };
			return getIndex(dim);
		}

		// TODO: Speed this the C up
		int getIndex(std::vector<int> dim) {
			std::reverse(dim.begin(), dim.end());

			int n = 0;
			for (int i = 0; i < dim.size(); i++) {
				int sum = 1;
				for (int j = 0; j < i; j++) {
					sum *= dimensionsReversed[j];
				}
				n += sum * dim[i];
			}

			return n;
		}

		// This might solve things
		int getIndex2D(int x, int y) {
			return dimensions.at(0) * x + y;
		}

		int size() {
			return m_size;
		}

		void fill(T value) {
			for (int i = 0; i < m_size; i++) {
				//*(m_data + i * sizeof(T)) = value;
				this->operator[](i) = value;
			}
			copyMemoryToDevice();
			//std::cout << "[CONVEX] Filled array with " << m_size << " elements of value " << value << std::endl;
		}

		void print() {
			//std::cout << dimensions[0] << ", " << dimensions[1] << std::endl;

			std::cout << "[";
			for (int i = 0; i < dimensions[0]; i++) {
				if (i != 0) std::cout << " ";
				std::cout << "[";
				for (int j = 0; j < dimensions[1]; j++) {
					std::cout << at(i, j);
					if (j != dimensions[1] - 1) std::cout << ", ";
				}
				std::cout << "]";
				if (i != dimensions[0] - 1) std::cout << ", \n";
			}
			std::cout << "]" << std::endl;
		}

		void copyMemoryToHost() {
			//std::cout << "[CONVEX] Copying " << m_size * sizeof(T) << " bytes to host" << std::endl;
			cudaError_t res = cudaMemcpy(m_hostData, m_deviceData, m_size * sizeof(T), cudaMemcpyDeviceToHost);
			if (res != cudaSuccess) {
				throw std::runtime_error("Failed to copy to host memory");
			}
		}

		void copyMemoryToDevice() {
			//std::cout << "[CONVEX] Copying " << m_size * sizeof(T) << " bytes to device" << std::endl;
			cudaError_t res = cudaMemcpy(m_deviceData, m_hostData, m_size * sizeof(T), cudaMemcpyHostToDevice);
			if (res != cudaSuccess) {
				std::cout << "[CONVEX ERROR] CUDA error " << res << std::endl;
				throw std::runtime_error("Failed to copy to device memory");
			}
		}

		void freeMemory() {
			free(m_hostData);
			cudaFree(m_deviceData);
		}

		// TODO: improve error handling for dims that have an i < m_size, but have invalid coordinates eg [0, 3] in a 2x2 matrix
		template <typename... Args>
		T& at(Args... indexes) {
			int i = getIndex(indexes...);
			if (i <= m_size) return this->operator[](i);
			else {
				std::cout << "[ERROR] Index " << i << " exceeds array bounds" << std::endl;
				throw std::out_of_range("Index exceeds array bounds");
			}
		}

		T& at(std::vector <int> dim) {
			int i = getIndex(dim);
			if (i <= m_size) return this->operator[](i);
			else {
				std::cout << "[ERROR] Index " << i << " exceeds array bounds" << std::endl;
				throw std::out_of_range("Index exceeds array bounds");
			}
		}

		T& at2D(int x, int y) {
			int i = getIndex2D(x, y);
			if (i <= m_size) return this->operator[](i);
			else {
				std::cout << "[ERROR] Index " << i << " exceeds array bounds" << std::endl;
				throw std::out_of_range("Index exceeds array bounds");
			}
		}

		//This intentionally doesn't execute the copyMemoryToDevice, since it would be very inefficient
		T& operator[] (int i) {
			if (i < m_size && i >= 0) return m_hostData[i];
			else throw std::out_of_range("Index exceeds array bounds");
		}
		const T& operator[] (int i) const {
			if (i < m_size && i >= 0) return m_hostData[i];
			else throw std::out_of_range("Index exceeds array bounds");
		}
	};

	Matrix <double> matrixMultiply2D(Matrix <double> *a, Matrix <double> *b);

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
		double train(ImageClassDataset* _dataset);
		void trainSequence(ImageClassDataset* _dataset, int _epochs, const char* _path);
		double assess(ImageClassDataset* _dataset);

		std::vector <std::vector <double>> matrixMultiply(std::vector<std::vector <double>>* _matrixA, std::vector<std::vector <double>>* _matrixB);

		void freeMemory();

		//std::ofstream* serialise(std::ofstream* _outputStream, bool _swapEndianness = false);
		//void serialise(const char * _path, bool _swapEndianness = false);

		//std::ifstream* deserialise(std::ifstream* _inputStream, bool _swapEndianness = false);
		//void deserialise(const char* _path, bool _swapEndianness = false);

		double normalise(double _input);
		double deriveNormalise(double _input);

		double* getWeight(int _n1Layer, int _n1Neuron, int _n2Layer, int _n2Neuron);
	};
}