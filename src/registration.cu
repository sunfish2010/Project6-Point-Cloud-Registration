/**
* @file      rasterize.cu
* @brief     CUDA-accelerated rasterization pipeline.
* @authors   Skeleton code: Yining Karl Li, Kai Ninomiya, Shuai Shao (Shrek)
* @date      2012-2016
* @copyright University of Pennsylvania & STUDENT
*/

#include <cmath>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/random.h>
#include <util/checkCUDAError.h>
#include "registrationTools.h"
#include "registration.h"
#include "kdtree.h"
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <thrust/reduce.h>
#include <thrust/device_vector.h>

#include <thrust/execution_policy.h>
#include <util/svd3.h>

#ifndef imax
#define imax(a, b) (((a) > (b)) ? (a) : (b))
#endif

#ifndef imin
#define imin(a, b) (((a) < (b)) ? (a) : (b))
#endif

#define EXHAUSTIVE 1
#define KDTREE 1

#define blockSize 128
#define scene_scale 100.0f
#define threadsPerBlock(blockSize)

template<typename T>
__host__ __device__

void swap(T &a, T &b) {
	T tmp(a);
	a = b;
	b = tmp;
}

static int numObjects;

static glm::vec3 *dev_pos_fixed = NULL;
static glm::vec3 *dev_pos_rotated = NULL;
static glm::vec3 *dev_pos_corr = NULL;
static glm::vec3 *dev_pos_rotated_centered = NULL;
static glm::mat3 *dev_w = NULL;
static Node *dev_kd = NULL;

//static cudaEvent_t start, stop;
/**
* Kernel that writes the image to the OpenGL PBO directly.
*/
/******************
* copyPtsToVBO *
******************/

/**
* Copy the boid positions into the VBO so that they can be drawn by OpenGL.
*/
__global__ void kernCopyPositionsToVBO(int N, int offset, glm::vec3 *pos, float *vbo) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);

	float c_scale = -1.0f / scene_scale;

	if (index < N) {
		vbo[4 * (index + offset) + 0] = pos[index].x * c_scale;
		vbo[4 * (index + offset) + 1] = pos[index].y * c_scale;
		vbo[4 * (index + offset) + 2] = pos[index].z * c_scale;
		vbo[4 * (index + offset) + 3] = 1.0f;
	}
}

__global__ void kernCopyColorsToVBO(int N, int offset, glm::vec3 color, float *vbo) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);

	if (index < N) {
		vbo[4 * (index + offset) + 0] = color.x;
		vbo[4 * (index + offset) + 1] = color.y;
		vbo[4 * (index + offset) + 2] = color.z;
		vbo[4 * (index + offset) + 3] = 1.0f;
	}
}

/**
* Wrapper for call to the kernCopyboidsToVBO CUDA kernel.
*/
void copyPointsToVBO(float *vbodptr_positions, float *vbodptr_velocities) {
	dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

	kernCopyPositionsToVBO << <fullBlocksPerGrid, blockSize >> >(numObjects, 0, dev_pos_fixed, vbodptr_positions);
	checkCUDAError("copyPositionsFixed failed!");

	kernCopyColorsToVBO << <fullBlocksPerGrid, blockSize >> >(numObjects, 0, glm::vec3(1.0f, 1.0f, 1.0f),
		vbodptr_velocities);
	checkCUDAError("copyColorsFixed failed!");

	kernCopyPositionsToVBO << < fullBlocksPerGrid, blockSize >> >(numObjects, numObjects,
		dev_pos_rotated, vbodptr_positions);
	checkCUDAError("copyPositionsRotated failed!");

	kernCopyColorsToVBO << < fullBlocksPerGrid, blockSize >> >(numObjects, numObjects, glm::vec3(0.3f, 0.9f, 0.3f),
		vbodptr_velocities);
	checkCUDAError("copyColorsRotated failed!");


	cudaDeviceSynchronize();
}


__global__ void kernInitializePosArray(int N, glm::vec3 *pos, float scale) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index < N) {
		pos[index].x *= scale;
		pos[index].y *= scale;
		pos[index].z *= scale;
	}
}


__global__ void transformPoints(int N, glm::vec3 *pos_in, glm::vec3 *pos_out, glm::mat4 transformation) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index < N) {
		pos_out[index] = glm::vec3(transformation * glm::vec4(pos_in[index], 1.0f));
	}
}


glm::mat4 constructTransformationMatrix(const glm::vec3 &translation, const glm::vec3& rotation, const glm::vec3& scale) {
	glm::mat4 translation_matrix = glm::translate(glm::mat4(), translation);
	glm::mat4 rotation_matrix = glm::rotate(glm::mat4(), rotation.x, glm::vec3(1, 0, 0));
	rotation_matrix *= glm::rotate(glm::mat4(), rotation.y, glm::vec3(0, 1, 0));
	rotation_matrix *= glm::rotate(glm::mat4(), rotation.z, glm::vec3(0, 0, 1));
	glm::mat4 scale_matrix = glm::scale(glm::mat4(), scale);
	return translation_matrix* rotation_matrix * scale_matrix;
}

glm::mat4 constructTranslationMatrix(const glm::vec3 &translation) {
	glm::mat4 translation_matrix = glm::translate(glm::mat4(), translation);
	return translation_matrix;
}

/**
* Called once at the beginning of the program to allocate memory.
*/
void registrationInitGPU(const std::vector<glm::vec3>& pts) {
	numObjects = (int)pts.size();
	dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
	cudaMalloc((void**)&dev_pos_fixed, numObjects * sizeof(glm::vec3));
	cudaMalloc((void**)&dev_pos_rotated, numObjects * sizeof(glm::vec3));
	cudaMalloc((void**)&dev_pos_corr, numObjects * sizeof(glm::vec3));
	cudaMalloc((void**)&dev_pos_rotated_centered, numObjects * sizeof(glm::vec3));
	cudaMalloc((void**)&dev_w, numObjects * sizeof(glm::mat3));

	checkCUDAError("registration Init");

	cudaMemcpy(dev_pos_fixed, &pts[0], numObjects * sizeof(glm::vec3), cudaMemcpyHostToDevice);
	checkCUDAError("pos_fixed Memcpy");

#if KDTREE
	cudaMalloc((void**)&dev_kd, numObjects * sizeof(Node));
	KDTree(pts);
	std::vector<Node> tree = pts.getTree();
	cudaMemcpy(dev_kd, &tree[0], numObjects * sizeof(Node), cudaMemcpyHostToDevice);
#endif

	kernInitializePosArray << <fullBlocksPerGrid, blockSize >> > (numObjects, dev_pos_fixed, scene_scale);

	glm::mat4 transformation = constructTransformationMatrix(glm::vec3(1.0f, 0.0f, 0.0f),
		glm::vec3(0.4f, 0.4f, -0.2f), glm::vec3(1.0f, 1.0f, 1.0f));

	transformPoints << <fullBlocksPerGrid, blockSize >> > (numObjects, dev_pos_fixed,
		dev_pos_rotated, transformation);
}


__global__ void findNearestNeighborExhaustive(int N, const glm::vec3 *source, const glm::vec3 *target, glm::vec3 *corr) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index < N) {
		glm::vec3 pt = source[index];
		float d_closest = glm::distance(target[0], pt);
		int i = 0;
		for (int j = 1; j < N; j++) {
			float d = glm::distance(target[j], pt);
			if (d < d_closest) {
				d_closest = d;
				i = j;
			}
		}
		corr[index] = target[i];
	}
}


__device__ float calculateHyperPlaneDist(const glm::vec3& pt1, const glm::vec3& pt2, int axis){
	if (axis == 0) return pt1.x - pt2.x;
	else if (axis == 1) return pt1.y - pt2.y;
	else return pt1.z - pt2.z;
}


__global__ void findNearestNeighborKDTree(int N, const glm::vec3 *source, const Node *tree, glm::vec3 *corr){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(index < N{
		glm::vec3 pt = source[index];
		float d_closest = glm::distance(tree[0].data, pt);
		bool explored = false;
		float hyper_dist = calculateHyperPlaneDist(pt, tree[0].data, tree[0].axis);
		int curr_node = hyper_dist < 0 ? tree[0].left, tree[0].right;
		int closest_node = 0;
		while(1){
			// explore current node & below
			while(curr_node != -1){
				float d = glm::distance(tree[curr_node].data, pt);
				if (d < d_closest){
					d_closest = d;
					closest_node = curr_node;
				}
				hyper_dist = calculateHyperPlaneDist(pt, tree[curr_node].data, tree[curr_node].axis);
				curr_node = hyper_dist < 0 ? tree[curr_node].left, tree[curr_node].right;

			}
			if(explored) break;
			else{
				int parent = tree[closest_node].parent;
				if (parent == -1) break;
				hyper_dist = calculateHyperPlaneDist(pt, tree[parent].data, tree[parent].axis);
				if (abs(hyper_dist) < d_closest){
					curr_node = hyper_dist < 0 ? tree[parent].eft, tree[parent].right;
				}else break;
			}


		}
		corr[index] = tree[closest_node].data;
	}
}


__global__ void translatePts(int N, glm::vec3* pos_in, glm::vec3* pos_out, glm::mat4 translation) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index < N) {
		pos_out[index] = glm::vec3(translation * glm::vec4(pos_in[index], 1.f));
	}
}


__global__ void calculateW(int N, glm::vec3* pos_rotated, glm::vec3* pos_cor, glm::mat3* w) {
	int index =  (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index < N) {
		w[index] = glm::outerProduct(pos_rotated[index], pos_cor[index]);
		// w[index] = glm::mat3(pos_rotated[index] * pos_cor[index].x,
			// pos_rotated[index] * pos_cor[index].y, pos_rotated[index] * pos_cor[index].z);
	}
}


// __global__ void calculateSVDWrapper(glm::mat3& w, glm::mat3& S, glm::mat3& U, glm::mat3 &V) {
// 	svd(w[0][0], w[0][1], w[0][2], w[1][0], w[1][1], w[1][2], w[2][0], w[2][1], w[2][2],
// 		U[0][0], U[0][1], U[0][2], U[1][0], U[1][1], U[1][2], U[2][0], U[2][1], U[2][2],
// 		S[0][0], S[0][1], S[0][2], S[1][0], S[1][1], S[1][2], S[2][0], S[2][1], S[2][2],
// 		V[0][0], V[0][1], V[0][2], V[1][0], V[1][1], V[1][2], V[2][0], V[2][1], V[2][2]);
// }




/**
* Perform point cloud registration.
*/
void registration() {
	dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
#if EXHAUSTIVE
	findNearestNeighborExhaustive << <fullBlocksPerGrid, blockSize >> > (numObjects, dev_pos_rotated,
		dev_pos_fixed, dev_pos_corr);
#elif KDTREE
	findNearestNeighborKDTree << <fullBlocksPerGrid, blockSize >> > (numObjects, dev_pos_rotated,
		dev_kd , dev_pos_corr);
#endif

	checkCUDAError("Find nearest Neighbor");

	// ICP Algorithm based on http://ais.informatik.uni-freiburg.de/teaching/ss11/robotics/slides/17-icp.pdf

	// calculate mean of two point clouds using stream compaction
	thrust::device_ptr<glm::vec3> thrust_pos_corr(dev_pos_corr);
	thrust::device_ptr<glm::vec3> thrust_pos_rotated(dev_pos_rotated);

	glm::vec3 pos_corr_mean = thrust::reduce(thrust_pos_corr, thrust_pos_corr + numObjects,
		glm::vec3(0.f, 0.f, 0.f));
	pos_corr_mean /= numObjects;

	glm::vec3 pos_rotated_mean = glm::vec3(thrust::reduce(thrust_pos_rotated,
		thrust_pos_rotated + numObjects, glm::vec3(0.f, 0.f, 0.f))) ;
	pos_rotated_mean /= numObjects;

    glm::mat4 translation_matrix = constructTranslationMatrix(-pos_corr_mean);
	translatePts << <fullBlocksPerGrid, blockSize >> > (numObjects, dev_pos_corr, dev_pos_corr, translation_matrix);
		
    translation_matrix = constructTranslationMatrix(-pos_rotated_mean);
	translatePts << <fullBlocksPerGrid, blockSize >> > (numObjects, dev_pos_rotated, dev_pos_rotated_centered, translation_matrix);
		
	checkCUDAError("Translating Pts");

	calculateW << < fullBlocksPerGrid, blockSize >> > (numObjects, dev_pos_rotated_centered, dev_pos_corr, dev_w);
	thrust::device_ptr<glm::mat3> thrust_w(dev_w);
	glm::mat3 W = thrust::reduce(thrust_w, thrust_w + numObjects, glm::mat3(0.f));
	checkCUDAError("Calculated W");

	glm::mat3 S, U, V;

    // could not get gpu version working
	// calculateSVDWrapper <<<1, 1 >>> (W, S, U, V);
	// checkCUDAError("SVD W");
    // the faster implementation creates error that makes det non-one
    svd(W[0][0], W[0][1], W[0][2], W[1][0], W[1][1], W[1][2], W[2][0], W[2][1], W[2][2],
     U[0][0], U[0][1], U[0][2], U[1][0], U[1][1], U[1][2], U[2][0], U[2][1], U[2][2],
     S[0][0], S[0][1], S[0][2], S[1][0], S[1][1], S[1][2], S[2][0], S[2][1], S[2][2],
     V[0][0], V[0][1], V[0][2], V[1][0], V[1][1], V[1][2], V[2][0], V[2][1], V[2][2]);

	glm::mat3 R = glm::transpose(U) * V;
	float det_R = glm::determinant(R);
	//glm::mat4 scale_matrix = glm::scale(glm::mat4(), glm::vec3(1.f / det_R, 1.f / det_R, 1.f / det_R));
	glm::vec3 t = pos_corr_mean - R * pos_rotated_mean;
	glm::mat4 T = glm::translate(glm::mat4(), t);
	//glm::mat4 transformation = T * glm::mat4(R) * scale_matrix;
    glm::mat4 transformation = T * glm::mat4(R);
	float det_transform = glm::determinant(transformation);
	transformPoints << < fullBlocksPerGrid, blockSize >> > (numObjects, dev_pos_rotated, dev_pos_rotated, transformation);
	checkCUDAError("Transforming Pts");
}

/**
* Called once at the end of the program to free CUDA memory.
*/
void registrationFree() {

	// deconstruct primitives attribute/indices device buffer

	cudaFree(dev_pos_rotated);
	cudaFree(dev_pos_fixed);
	cudaFree(dev_pos_rotated_centered);
	cudaFree(dev_pos_corr);
	cudaFree(dev_w);
	cudaFrree(dev_kd);

	dev_pos_fixed = NULL;
	dev_pos_rotated = NULL;
	dev_pos_corr = NULL;
	dev_pos_rotated_centered = NULL;
	dev_w = NULL;

	checkCUDAError("registration Free");
}


template <typename T>
T calculate_vector_mean(std::vector<T> input){
	T sum;
	for (auto &element: input){
		sum += element;
	}
	return sum /= input.size();
};


std::vector<glm::vec3> registration_init_cpu(std::vector<glm::vec3> &input){
    glm::mat4 transformation = constructTransformationMatrix(glm::vec3(1.0f, 0.0f, 0.0f),
        glm::vec3(0.4f, 0.4f, -0.2f), glm::vec3(1.0f, 1.0f, 1.0f));
    std::vector<glm::vec3>result(input.size(), glm::vec3(0.f, 0.f, 0.f));
    for (int i = 0; i < input.size(); ++i){
        result[i] = glm::vec3(transformation * glm::vec4(input[i], 1.0f));
    }
	return result;
}

// skeleton code for cpu_step; no display, just for performance comparison
void registration_cpu(std::vector<glm::vec3>& target, std::vector<glm::vec3>& source){
	int numPts = target.size();

	std::vector<glm::vec3> corr(numPts, glm::vec3(0.f, 0.f, 0.f));

	for (int k = 0; k < numPts; k++){
		auto best_dist = glm::distance(source[k], target[0]);
		int i = 0;
		for (int j = 1; j < numPts; j++){
			auto d = glm::distance(source[k], target[j]);
			if ( d < best_dist){
				best_dist = d;
				i = j;
			}
		}
		corr[k] = target[i];
	}


	glm::vec3 mean_corr = calculate_vector_mean(corr);
	glm::vec3 mean_source = calculate_vector_mean(source);

	std::vector<glm::vec3> source_centered = source;

	for (int i = 0; i < numPts; i++){
		source_centered[i] = source[i] - mean_source;
		corr[i] -= mean_corr;
	}

	// calculate w
	std::vector<glm::mat3> w(numPts, glm::mat3(glm::vec3(0.f, 0.f, 0.f), glm::vec3(0.f, 0.f, 0.f), glm::vec3(0.f, 0.f, 0.f)));
	for (int i = 0; i < numPts; i++){
		w[i] = glm::outerProduct(source_centered[i], corr[i]);
	}

	glm::mat3 W = calculate_vector_mean(w);
	W *= numPts;

	glm::mat3 S, U, V;

	svd(W[0][0], W[0][1], W[0][2], W[1][0], W[1][1], W[1][2], W[2][0], W[2][1], W[2][2],
		U[0][0], U[0][1], U[0][2], U[1][0], U[1][1], U[1][2], U[2][0], U[2][1], U[2][2],
		S[0][0], S[0][1], S[0][2], S[1][0], S[1][1], S[1][2], S[2][0], S[2][1], S[2][2],
		V[0][0], V[0][1], V[0][2], V[1][0], V[1][1], V[1][2], V[2][0], V[2][1], V[2][2]);

	glm::mat3 R = glm::transpose(U) * V;
	glm::vec3 t = mean_corr - R * mean_source;
	glm::mat4 T = glm::translate(glm::mat4(), t);
	glm::mat4 transformation = T * glm::mat4(R);

	for (int i = 0; i < numPts; i++){
		source[i] = glm::vec3(transformation * glm::vec4(source[i], 1.0f));
	}

}