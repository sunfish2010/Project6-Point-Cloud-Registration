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

void copyPointsToVBO(float *vbodptr_positions, float *vbodptr_velocities);

void registrationInit(const vector<glm::vec4>& pts);
void registration(int method);
void registrationFree();
