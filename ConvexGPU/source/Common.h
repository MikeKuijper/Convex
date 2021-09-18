#pragma once
#include <vector>
#include <iostream>
#include <random>
#include <algorithm>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include <chrono>

double randomNumber(double _min, double _max) {
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dis(_min, _max);
	return dis(gen);
}

template <class variable>
void endianSwap(variable *objp) {
	unsigned char *memp = reinterpret_cast<unsigned char*>(objp);
	std::reverse(memp, memp + sizeof(variable));
}

template <class variableType>
void writeVariable(variableType* _variable, std::ofstream* _fs, bool _swapEndianness) {
	variableType &temp = *_variable;
	if (_swapEndianness == true) endianSwap(&temp);
	_fs->write(reinterpret_cast<char*> (&temp), sizeof(variableType));
}

template <class variableType>
void writeCustom(variableType* _variable, unsigned int _size, std::ofstream* _fs, bool _swapEndianness) {
	variableType &temp = *_variable;
	if (_swapEndianness == true) endianSwap(&temp);
	_fs->write(reinterpret_cast<char*> (&temp), _size);
}

template <class variableType>
variableType readVariable(variableType* _variable, std::ifstream* _fs, bool _swapEndianness) {
	_fs->read(reinterpret_cast<char*> (_variable), sizeof(*_variable));
	if (_swapEndianness) endianSwap(_variable);
	return *_variable;
}

template <class variableType>
variableType readCustom(variableType* _variable, unsigned int _size, std::ifstream* _fs, bool _swapEndianness) {
	_fs->read(reinterpret_cast<char*> (_variable), _size);
	if (_swapEndianness) endianSwap(_variable);
	return *_variable;
}

template <class number>
void writeVector(std::vector <number>* _vector, std::ofstream* _fs, bool _swapEndianness) {
	unsigned int vectorSize = (unsigned int)_vector->size();
	writeVariable(&vectorSize, _fs, _swapEndianness);

	for (int i = 0; i < vectorSize; i++) {
		writeVariable(&_vector->at(i), _fs, _swapEndianness);
	}
}

template <class number>
void writeVector(std::vector <std::vector <number>>* _vector, std::ofstream* _fs, bool _swapEndianness) {
	unsigned int vectorSize = (unsigned int)_vector->size();
	writeVariable(&vectorSize, _fs, _swapEndianness);

	for (int i = 0; i < vectorSize; i++) {
		writeVector(&_vector->at(i), _fs, _swapEndianness);
	}
}

template <class number>
void writeVector(std::vector <std::vector <std::vector <number>>>* _vector, std::ofstream* _fs, bool _swapEndianness) {
	unsigned int vectorSize = (unsigned int)_vector->size();
	writeVariable(&vectorSize, _fs, _swapEndianness);

	for (int i = 0; i < vectorSize; i++) {
		writeVector(&_vector->at(i), _fs, _swapEndianness);
	}
}

template <class Matrix>
void writeMatrixVector(std::vector <Matrix*>* _vector, std::ofstream* _fs, bool _swapEndianness) {
	unsigned int vectorSize = (unsigned int)_vector->size();
	writeVariable(&vectorSize, _fs, _swapEndianness);

	for (int i = 0; i < vectorSize; i++) {
		_vector->at(i)->serialise(_fs, _swapEndianness);
	}
}

template <class Matrix>
void readMatrixVector(std::vector <Matrix*>* _vector, std::ifstream* _fs, bool _swapEndianness) {
	unsigned int vectorSize;
	readVariable(&vectorSize, _fs, _swapEndianness);

	for (int i = 0; i < vectorSize; i++) {
		Matrix* m = new Matrix(_fs, _swapEndianness);
		_vector->push_back(m);
	}
}

/* template <class Matrix>
void writeMatrixVector(std::vector <Matrix>* _vector, std::ofstream* _fs, bool _swapEndianness) {
	unsigned int vectorSize = (unsigned int)_vector->size();
	writeVariable(&vectorSize, _fs, _swapEndianness);

	for (int i = 0; i < vectorSize; i++) {
		_vector->at(i)->serialise(_fs, _swapEndianness);
	}
}

template <class Matrix>
void readMatrixVector(std::vector <Matrix>* _vector, std::ifstream* _fs, bool _swapEndianness) {
	unsigned int vectorSize;
	readVariable(&vectorSize, _fs, _swapEndianness);

	for (int i = 0; i < vectorSize; i++) {
		Matrix m(_fs, _swapEndianness);
		_vector->push_back(m);
	}
} */

template <typename number>
void readVector(std::vector <number>* _vector, std::ifstream* _fs, bool _swapEndianness) {
	unsigned int vectorSize;
	readVariable(&vectorSize, _fs, _swapEndianness);

	for (int i = 0; i < vectorSize; i++) {
		number var;
		_vector->push_back(readVariable(&var, _fs, _swapEndianness));
	}
}

template <typename number>
void readVector(std::vector <std::vector <number>>* _vector, std::ifstream* _fs, bool _swapEndianness) {
	unsigned int vectorSize;
	readVariable(&vectorSize, _fs, _swapEndianness);

	for (int i = 0; i < vectorSize; i++) {
		std::vector <number> vector;
		readVector(&vector, _fs, _swapEndianness);
		_vector->push_back(vector);
	}
}

template <class number>
std::vector<number> subVector(std::vector <number>* _vector, unsigned int _i, unsigned int _j) {
	auto first = _vector->begin() + _i;
	auto last = _vector->begin() + _j + 1;
	std::vector <number> v(first, last);
	return v;
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

template <class Number>
void printVector(std::vector <Number>* _vector) {
	std::cout << "[";
	for (int i = 0; i < _vector->size(); i++) {
		if (i != 0) std::cout << ", ";
		std::cout << _vector->at(i);
	}
	std::cout << "]" << std::endl;
}

template <class Vector>
void printVector(std::vector <std::vector<Vector>>* _vector) {
	std::cout << "[";
	for (int i = 0; i < _vector->size(); i++) {
		if (i != 0) std::cout << ", ";
		printVector(&_vector[i]);
	}
	std::cout << "]" << std::endl;
}
