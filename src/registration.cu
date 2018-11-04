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
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <thrust/remove.h>
#include <thrust/execution_policy.h>


#ifndef imax
#define imax(a, b) (((a) > (b)) ? (a) : (b))
#endif

#ifndef imin
#define imin(a, b) (((a) < (b)) ? (a) : (b))
#endif

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

static glm::vec4 *dev_pos_fixed = NULL;
static glm::vec4 *dev_pos_rotated = NULL;

//static cudaEvent_t start, stop;
/**
* Kernel that writes the image to the OpenGL PBO directly.
*/
/******************
* copyBoidsToVBO *
******************/

/**
* Copy the boid positions into the VBO so that they can be drawn by OpenGL.
*/
__global__ void kernCopyPositionsToVBO(int N, int offset, glm::vec4 *pos, float *vbo) {
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

    kernCopyPositionsToVBO <<<fullBlocksPerGrid, blockSize >>>(numObjects, 0, dev_pos_fixed, vbodptr_positions);
	checkCUDAError("copyPositionsFixed failed!");

	kernCopyColorsToVBO <<<fullBlocksPerGrid, blockSize >>>(numObjects, 0, glm::vec3(1.0f, 1.0f, 1.0f),
	        vbodptr_velocities);
    checkCUDAError("copyColorsFixed failed!");

	kernCopyPositionsToVBO <<< fullBlocksPerGrid, blockSize >>>(numObjects, numObjects,
	        dev_pos_rotated, vbodptr_positions);
    checkCUDAError("copyPositionsRotated failed!");

    kernCopyColorsToVBO <<< fullBlocksPerGrid, blockSize >>>(numObjects, numObjects, glm::vec3(0.3f, 0.9f, 0.3f),
            vbodptr_velocities);
    checkCUDAError("copyColorsRotated failed!");


    cudaDeviceSynchronize();
}


 __global__ void kernInitializePosArray(int N, glm::vec4 *pos, float scale){
     int index = (blockIdx.x * blockDim.x) + threadIdx.x;
     if (index < N){
         pos[index].x *= scale;
         pos[index].y *= scale;
         pos[index].z *= scale;
     }
 }


 __global__ void kernInitializePosArrayRotated(int N, glm::vec4 *pos_in, glm::vec4 *pos_out, glm::mat4 transformation){
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index < N){
        pos_out[index] = transformation * pos_in[index];
    }
}


glm::mat4 constructTransformationMatrix(const glm::vec3 &translation,const glm::vec3& rotation,const glm::vec3& scale){
    glm::mat4 translation_matrix = glm::translate(glm::mat4(), translation);
    glm::mat4 rotation_matrix = glm::rotate(glm::mat4(), rotation.x, glm::vec3(1, 0, 0));
    rotation_matrix *= glm::rotate(glm::mat4(), rotation.y, glm::vec3(0, 1, 0));
    rotation_matrix *= glm::rotate(glm::mat4(), rotation.z, glm::vec3(0, 0, 1));
    glm::mat4 scale_matrix = glm::scale(glm::mat4(), scale);
    return translation_matrix* rotation_matrix * scale_matrix;
}

/**
* Called once at the beginning of the program to allocate memory.
*/
void registrationInit(const std::vector<glm::vec4>& pts) {
    numObjects = (int)pts.size();
    dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
    cudaMalloc((void**) &dev_pos_fixed, numObjects * sizeof(glm::vec4));
    cudaMalloc((void**) &dev_pos_rotated, numObjects * sizeof(glm::vec4));

    checkCUDAError("registration Init");

	cudaMemcpy(dev_pos_fixed, &pts[0], numObjects * sizeof(glm::vec4), cudaMemcpyHostToDevice);
	checkCUDAError("pos_fixed Memcpy");

    kernInitializePosArray <<<fullBlocksPerGrid, blockSize>>> (numObjects, dev_pos_fixed, scene_scale);

    glm::mat4 transformation = constructTransformationMatrix(glm::vec3(5.0f, 0.0f, 0.0f),
            glm::vec3(0.4f, 0.6f, -0.2f), glm::vec3(1.0f, 1.0f, 1.0f));

    kernInitializePosArrayRotated <<<fullBlocksPerGrid, blockSize>>> (numObjects, dev_pos_fixed,
            dev_pos_rotated, transformation);
}
//
///**
//* kern function with support for stride to sometimes replace cudaMemcpy
//* One thread is responsible for copying one component
//*/
//__global__
//void _deviceBufferCopy(int N, BufferByte *dev_dst, const BufferByte *dev_src, int n, int byteStride, int byteOffset,
//                       int componentTypeByteSize) {
//
//    // Attribute (vec3 position)
//    // component (3 * float)
//    // byte (4 * byte)
//
//    // id of component
//    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
//
//    if (i < N) {
//        int count = i / n;
//        int offset = i - count * n;    // which component of the attribute
//
//        for (int j = 0; j < componentTypeByteSize; j++) {
//
//            dev_dst[count * componentTypeByteSize * n
//                    + offset * componentTypeByteSize
//                    + j]
//
//                    =
//
//                    dev_src[byteOffset
//                            + count * (byteStride == 0 ? componentTypeByteSize * n : byteStride)
//                            + offset * componentTypeByteSize
//                            + j];
//        }
//    }
//
//
//}


/**
* Perform point cloud registration.
*/
void registration(int method) {

}

/**
* Called once at the end of the program to free CUDA memory.
*/
void registrationFree() {

    // deconstruct primitives attribute/indices device buffer

    cudaFree(dev_pos_rotated);
    cudaFree(dev_pos_fixed);

    dev_pos_fixed = NULL;
    dev_pos_rotated = NULL;

    checkCUDAError("registration Free");
}