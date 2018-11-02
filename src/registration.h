/**
* @file      rasterize.h
* @brief     CUDA-accelerated rasterization pipeline.
* @authors   Skeleton code: Yining Karl Li, Kai Ninomiya, Shuai Shao (Shrek)
* @date      2018-11-02
* @copyright University of Pennsylvania & STUDENT
*/

#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>

namespace tinygltf {
    class Scene;
}


void registrationInit(int width, int height);
void registrationSetBuffers(const tinygltf::Scene & scene);

void registration(uchar4 *pbo, const glm::mat4 & MVP, const glm::mat4 & MV, const glm::mat3 MV_normal, int primitive_type);
void registrationFree();
