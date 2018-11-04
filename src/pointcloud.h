//
// Created by Yu Sun on 11/2/18.
//

#pragma once

#include <vector>
#include <string>
#include <glm/glm.hpp>
#include <fstream>
#include <util/utilityCore.hpp>

class PointCloud{
public:
    PointCloud(std::string filename);
	PointCloud();
	~PointCloud();
    std::vector<glm::vec3> getPoints();
	int getNumPoints();

private:
    // x y z color
    std::vector<glm::vec3> points;
	int numPoints;
	std::ifstream fp_in;
};