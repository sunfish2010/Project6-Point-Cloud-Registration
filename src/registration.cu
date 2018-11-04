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
static glm::vec3 *dev_vel2;
static glm::vec3 *dev_vel1;

static cudaEvent_t start, stop;
/**
* Kernel that writes the image to the OpenGL PBO directly.
*/
/******************
* copyBoidsToVBO *
******************/

/**
* Copy the boid positions into the VBO so that they can be drawn by OpenGL.
*/
__global__ void kernCopyPositionsToVBO(int N, glm::vec4 *pos, float *vbo, float s_scale) {
    int index = threadIdx.x + (blockIdx.x * blockDim.x);

    float c_scale = -1.0f / s_scale;

    if (index < N) {
        vbo[4 * index + 0] = pos[index].x * c_scale;
        vbo[4 * index + 1] = pos[index].y * c_scale;
        vbo[4 * index + 2] = pos[index].z * c_scale;
        vbo[4 * index + 3] = 1.0f;
    }
}

__global__ void kernCopyVelocitiesToVBO(int N, glm::vec3 *vel, float *vbo, float s_scale) {
    int index = threadIdx.x + (blockIdx.x * blockDim.x);

    if (index < N) {
//        vbo[4 * index + 0] = vel[index].x + 0.3f;
//        vbo[4 * index + 1] = vel[index].y + 0.3f;
//        vbo[4 * index + 2] = vel[index].z + 0.3f;
//        vbo[4 * index + 3] = 1.0f;
        vbo[4 * index + 0] = 0.f;
        vbo[4 * index + 1] = 0.f;
        vbo[4 * index + 2] = 0.f;
        vbo[4 * index + 3] = 1.0f;
    }
}

/**
* Wrapper for call to the kernCopyboidsToVBO CUDA kernel.
*/
void copyPointsToVBO(float *vbodptr_positions, float *vbodptr_velocities) {
    dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

    kernCopyPositionsToVBO << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_pos_fixed, vbodptr_positions, scene_scale);
	checkCUDAError("copyPositions failed!");
	kernCopyVelocitiesToVBO << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_vel1, vbodptr_velocities, scene_scale);

    checkCUDAError("copyVelocities failed!");

    cudaDeviceSynchronize();
}


// __global__ void kernInitializePosArray(const std::vector<glm::vec4>& pts, int N, glm::vec3 *pos, float scale){
//     int index = (blockIdx.x * blockDim.x) + threadIdx.x;
//     if (index < N){
//         pos[index].x = pts[index].x * scale;
//         pos[index].y = pts[index].y * scale;
//         pos[index].z = pts[index].z * scale;
//     }
// }

/**
* Called once at the beginning of the program to allocate memory.
*/
void registrationInit(const std::vector<glm::vec4>& pts) {
    numObjects = (int)pts.size();
    dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
    cudaMalloc((void**) &dev_pos_fixed, numObjects * sizeof(glm::vec4));
    cudaMalloc((void**) &dev_pos_rotated, numObjects * sizeof(glm::vec4));
    cudaMalloc((void**) &dev_vel1, numObjects * sizeof(glm::vec3));
    cudaMalloc((void**) &dev_vel2, numObjects * sizeof(glm::vec3));
    checkCUDAError("registration Init");

	cudaMemcpy(dev_pos_fixed, &pts[0], numObjects * sizeof(glm::vec4), cudaMemcpyHostToDevice);
	checkCUDAError("pos_fixed Memcpy");

    //kernInitializePosArray <<<fullBlocksPerGrid, blockSize>>> (pts, numObjects, dev_pos_fixed, scene_scale);
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
    cudaFree(dev_vel2);
    cudaFree(dev_vel1);

    dev_pos_fixed = NULL;
    dev_pos_rotated = NULL;
    dev_vel1 = NULL;
    dev_vel2 = NULL;

    checkCUDAError("registration Free");
}