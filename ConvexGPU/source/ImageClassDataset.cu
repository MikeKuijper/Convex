#include "Common.h"
#include "ConvexGPU.h"

namespace ConvexGPU {
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

		m_size = 100; //Remove later

		for (int n = 1; n <= m_size; n++) {
			//std::vector <std::vector<double>> image;
			Matrix <double> image(m_rows, m_columns);
			for (int y = 0; y < m_rows; y++) {
				//std::vector <double> row;
				for (int x = 0; x < m_columns; x++) {
					unsigned char value;
					readVariable(&value, _inputStream, true);
					image.at(y, x) = value;
					//row.push_back((double)value);
				}
				//image.push_back(row);
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

		m_size = 100; //Remove later

		for (int n = 0; n < m_size; n++) {
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
		//std::cout << m_size << std::endl;
		for (int i = 0; i < m_size; i++) {
			//std::cout << i << std::endl;
			//m_images[i].print();
			m_imagesFlattened.push_back(m_images.at(i).flatten());
			//m_imagesFlattened[i].dimensions.push_back(1);
			m_imagesFlattened[i].dimensions.insert(m_imagesFlattened[i].dimensions.begin(), 1);

			m_imagesFlattened[i].dimensionsReversed = m_imagesFlattened[i].dimensions;
			std::reverse(m_imagesFlattened[i].dimensionsReversed.begin(), m_imagesFlattened[i].dimensionsReversed.end());
		}
		m_images.clear();
	}

	/*void ImageClassDataset::flattenMatrix() {
		for (int i = 0; i < m_size; i++) {
			//m_imagesFlattenedMatrix.push_back(flattenVector(m_images.at(i)));
		}
		m_images.clear();
	}*/

	void ImageClassDataset::shuffle() {
		unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
		std::shuffle(std::begin(m_imagesFlattened), std::end(m_imagesFlattened), std::default_random_engine(seed));
	}
}