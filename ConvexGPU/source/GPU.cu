#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <chrono>

//#include <math.h>
//#include <algorithm>

#include "Convex.h"
#include "ConvexGPU.h"

namespace ConvexGPU {
	__global__ void incrementKernel(double* data, int N) {
		*(data + threadIdx.x * sizeof(double)) = *(data + threadIdx.x * sizeof(double)) + 1;
	}

	__global__ void sigmoidKernel(double* data, int N) {
		double value = *(data + threadIdx.x * sizeof(double));
		*(data + threadIdx.x * sizeof(double)) = 1 / (1 + exp(-value));
	}

	__global__ void tanhKernel(double* data, int N) {
		double value = *(data + threadIdx.x * sizeof(double));
		*(data + threadIdx.x * sizeof(double)) = 2 / (1 + exp(-2 * value));
	}

	__global__ void matrixMultiplyKernel(double *a, double *b, double *c, int I, int aWidth, int bWidth) {
		//printf("THREADIDX: [%i, %i]\n", threadIdx.y, threadIdx.x);
		//printf("BLOCKDIM:  [%i, %i]\n", blockDim.x, blockDim.y);
		//int row = blockIdx.y * blockDim.y + threadIdx.y;
		//int col = blockIdx.x * blockDim.x + threadIdx.x;
		int row = threadIdx.y;
		int col = threadIdx.x;

		if (row < blockDim.y && col < blockDim.x) {
			double elementSum = 0;

			for (int i = 0; i < I; i++) {
				int indexA = row * aWidth + i;
				double elementA = a[indexA];
				int indexB = i * bWidth + col;
				double elementB = b[indexB];

				//printf("[%i, %i]: A[%i, %i](%i) = %f, B[%i, %i](%i) = %f\n", row, col, row, i, indexA, elementA, i, col, indexB, elementB);

				elementSum += elementA * elementB;
			}

			//printf("[%i, %i]: %f [%i, %i]\n", row, col, elementSum, N, M);
			//printf("[%i, %i]: %f\n", row, col, elementSum);
			c[row * blockDim.x + col] = elementSum;
		}
	}

	//TODO allow for different data types eg int, float, long
	//TODO fix code for larger matrixes
	Matrix <double> matrixMultiply2D(Matrix <double> *a, Matrix <double> *b) {
		if (a->dimensions[1] == b->dimensions[0]) {
			Matrix <double> resultMatrix(a->dimensions[0], b->dimensions[1]);
			int ROWS = a->dimensions[0];
			int COLUMNS = b->dimensions[1];
			int ITERATIONS = a->dimensions[1];

			//std::cout << ROWS << "x" << COLUMNS << " with " << ITERATIONS << " iterations" << std::endl;

			dim3 threadsPerBlock(COLUMNS, ROWS);
			dim3 blocksPerGrid(1, 1);
			if (ROWS * COLUMNS > 512) {
				std::cout << "[CONVEX] Large matrixes" << std::endl;
				//threadsPerBlock.x = 32;
				//threadsPerBlock.y = 32;
				//blocksPerGrid.x = ceil(double(COLUMNS) / double(threadsPerBlock.x));
				//std::cout << "blocksPerGrid.x = " << blocksPerGrid.x << std::endl;
				//blocksPerGrid.y = ceil(double(ROWS) / double(threadsPerBlock.y));
				//std::cout << "blocksPerGrid.y = " << blocksPerGrid.y << std::endl;
			}

			matrixMultiplyKernel <<<blocksPerGrid, threadsPerBlock>>> (a->m_deviceData, b->m_deviceData, resultMatrix.m_deviceData, ITERATIONS, a->dimensions[1], b->dimensions[1]);
			cudaDeviceSynchronize();

			resultMatrix.copyMemoryToHost();

			return resultMatrix;
		}
		else {
			std::cout << "[ERROR] Matrixes are incompatible for multiplication" << std::endl;
			throw std::invalid_argument("Matrixes aren't compatible for multiplication");
		}
	}

	void testSetupMatrixMultiply() {
		//int N = 32;
		Matrix <double> a(1, 2);
		Matrix <double> b(2, 2);

		a.fill(2);
		a.print();

		b.fill(2);
		b.print();

		auto start = std::chrono::high_resolution_clock::now();
		Matrix <double> c = matrixMultiply2D(&a, &b);
		auto stop = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

		std::cout << "[CONVEX] Multiplication took " << duration.count() << " microseconds; it could do " << 1000000.0f / duration.count() << " such calculations per second" << std::endl;

		c.print();
		std::cout << c[0] << std::endl;
		std::cout << c[c.m_size - 1] << std::endl;
	}

	void testSetupIncrement() {
		std::cout << "[CONVEX] Testing setup" << std::endl;

		Matrix <double> M(3);
		std::cout << "[CONVEX] Matrix initialised" << std::endl;
		M.fill(0);

		std::vector <double> before;
		std::vector <double> after;

		before.push_back(M[0]);
		before.push_back(M[1]);
		before.push_back(M[2]);

		//std::cout << M.dimensions[0] << std::endl;
		incrementKernel <<<1, M.dimensions[0]>>> (M.m_deviceData, M.dimensions[0]);
		cudaDeviceSynchronize();

		after.push_back(M[0]);
		after.push_back(M[1]);
		after.push_back(M[2]);

		std::cout << "[CONVEX] Setup test completed" << std::endl;

		if (before[0] == 0 && before[1] == 0 && before[2] == 0 && after[0] == 1 && after[1] == 1 && after[2] == 1) {
			std::cout << "[CONVEX] Setup behaves as expected" << std::endl;
		}
		else {
			std::cout << "[CONVEX] An error occured while testing setup" << std::endl;
		}
	}
}