#pragma once
#include "Common.h"

namespace Convex {
	template<class T>
	class Matrix {
	public:
		std::vector <unsigned int> dimensions;
		std::vector <unsigned int> dimensionsReversed;
		T* m_hostData;
		unsigned int m_size;

		Matrix() {
			return;
		}

		~Matrix() {
			//cudaFree(m_deviceData);
			//freeMemory();
		}

		//TODO: allow for use with standard memory and AMD GPUs
		Matrix(std::vector<unsigned int> dim) {
			dimensions = dim;
			dimensionsReversed = dim;
			std::reverse(dimensionsReversed.begin(), dimensionsReversed.end());

			int size = 1;
			for (int i = 0; i < dimensionsReversed.size(); i++) {
				size *= dimensionsReversed.at(i);
			}

			m_hostData = (T*)malloc(sizeof(T) * size);
			m_size = size;
		}
		// TODO Fix duplicate code
		// TODO Check interchangability of above and below
		template <typename... Args>
		Matrix(Args... dims) {
			dimensions = { static_cast<unsigned int>(dims)... };
			dimensionsReversed = { static_cast<unsigned int>(dims)... };
			std::reverse(dimensionsReversed.begin(), dimensionsReversed.end());

			int size = 1;
			for (int i = 0; i < dimensions.size(); i++) {
				size *= dimensions.at(i);
			}

			m_hostData = (T*)malloc(sizeof(T) * size);
			m_size = size;
		}

		Matrix& flatten() {
			dimensions = { m_size };
			dimensionsReversed = { m_size };
			return *this;
		}

		void addDimension() {
			dimensions.insert(dimensionsReversed.begin(), 1);
			dimensionsReversed.push_back(1);
		}

		template <typename... Args>
		unsigned int getIndex(Args... indexes) {
			std::vector <unsigned int> dim = { static_cast<unsigned int>(indexes)... };
			return getIndex(dim);
		}
		// TODO: Speed this the C up
		unsigned int getIndex(std::vector<unsigned int> dim) {
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
		int getIndex2D(unsigned int x, unsigned int y) {
			return dimensions.at(0) * x + y;
		}

		unsigned int size() {
			return m_size;
		}

		Matrix& fill(T value) {
			for (unsigned int i = 0; i < m_size; i++) {
				this->operator[](i) = value;
			}
			return *this;
		}

		void print() {
			std::cout << "[";
			for (unsigned int i = 0; i < dimensions[0]; i++) {
				if (i != 0) std::cout << " ";
				std::cout << "[";
				for (unsigned int j = 0; j < dimensions[1]; j++) {
					std::cout << at(i, j);
					if (j != dimensions[1] - 1) std::cout << ", ";
				}
				std::cout << "]";
				if (i != dimensions[0] - 1) std::cout << ", \n";
			}
			std::cout << "]" << std::endl;
		}

		std::ofstream* serialise(std::ofstream* _stream, bool _swapEndianness = false) {
			writeVector(&dimensions, _stream, _swapEndianness);
			writeVariable(&m_size, _stream, _swapEndianness);
			writeCustom(m_hostData, m_size * sizeof(T), _stream, _swapEndianness);

			return _stream;
		}
		void serialise(const char * _path, bool _swapEndianness = false) {
			std::cout << "[CONVEX] Saving matrix to " << _path << "...";

			std::ofstream file;
			file.open(_path, std::ios::binary);

			serialise(&file, _swapEndianness);

			std::cout << " done" << std::endl;
			file.close();
		}

		std::ifstream* deserialise(std::ifstream* _stream, bool _swapEndianness = false) {
			dimensions.clear();
			readVector(&dimensions, _stream, _swapEndianness);
			unsigned int size;
			readVariable(&size, _stream, _swapEndianness);

			unsigned int trueSize = 1;
			for (int i = 0; i < dimensions.size(); i++) trueSize *= dimensions.at(i);

			if (size == trueSize) m_size = size;
			else {
				std::cerr << "[CONVEX ERROR] Input stream contains corrupted data" << std::endl;
				throw std::runtime_error("Corrupted input stream");
			}

			m_hostData = (T*)malloc(sizeof(T) * m_size);
			readCustom(m_hostData, m_size * sizeof(T), _stream, _swapEndianness);

			dimensionsReversed = dimensions;
			std::reverse(dimensionsReversed.begin(), dimensionsReversed.end());

			/*if (m_size != sizeof(m_hostData) / sizeof(T)) {
				std::cerr << "\n[CONVEX ERROR] Input stream contains corrupted data" << std::endl;
				throw std::runtime_error("Corrupted input stream");
			}
			copyMemoryToDevice();*/
			return _stream;
		}
		void deserialise(const char* _path, bool _swapEndianness = false) {
			std::cout << "[CONVEX] Loading matrix from " << _path << "...";

			std::ifstream file;
			file.open(_path, std::ios::binary);

			deserialise(&file, _swapEndianness);

			std::cout << " done" << std::endl;
			file.close();
		}

		void freeMemory() {
			free(m_hostData);
		}

		// TODO: improve error handling for dims that have an i < m_size, but have invalid coordinates eg [0, 3] in a 2x2 matrix
		template <typename... Args>
		T& at(Args... indexes) {
			auto i = getIndex(indexes...);
			if (i < m_size) return this->operator[](i);
			else {
				std::cerr << "[ERROR] Index " << i << " exceeds array bounds" << std::endl;
				throw std::out_of_range("Index exceeds array bounds");
			}
		}
		T& at(std::vector <unsigned int> dim) {
			auto i = getIndex(dim);
			if (i < m_size) return this->operator[](i);
			else {
				std::cerr << "[ERROR] Index " << i << " exceeds array bounds" << std::endl;
				throw std::out_of_range("Index exceeds array bounds");
			}
		}
		T& at2D(unsigned int x, unsigned int y) {
			int i = getIndex2D(x, y);
			if (i < m_size) return this->operator[](i); // changed from <= since it makes more sense; not tested yet
			else {
				std::cerr << "[ERROR] Index " << i << " exceeds array bounds" << std::endl;
				throw std::out_of_range("Index exceeds array bounds");
			}
		}

		//This intentionally doesn't execute copyMemoryToDevice, since it would be very inefficient
		T& operator[] (unsigned int i) {
			if (i < m_size) return m_hostData[i]; // changed from <= since it makes more sense; not tested yet
			else throw std::out_of_range("Index exceeds array bounds");
		}
		const T& operator[] (unsigned int i) const {
			if (i < m_size) return m_hostData[i]; // changed from <= since it makes more sense; not tested yet
			else throw std::out_of_range("Index exceeds array bounds");
		}
	};
}