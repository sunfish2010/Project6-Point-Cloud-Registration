//
// Created by Yu Sun on 11/2/18.
//

#pragma once

#include <vector>
#include <string>
#include <glm/glm.hpp>


class PointCloud{
public:
    PointCloud(std::string filename);
    std::vector<glm::vec4> getPoints();

private:
    // x y z color
    std::vector<glm::vec4> points;
};