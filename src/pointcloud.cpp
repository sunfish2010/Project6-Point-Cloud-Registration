//
// Created by Yu Sun on 11/2/18.
//

#include <iostream>
#include "pointcloud.h"
#include <cstring>
#include <glm/gtx/string_cast.hpp>
using namespace std;

PointCloud::PointCloud() {
}

PointCloud::~PointCloud() {
	this->points.clear();
	this->numPoints = 0;
}

PointCloud::PointCloud(std::string filename, int freq, char sep) {
    cout << "Reading Point Cloud From " << filename << "..." << endl;
    char* fname = (char*)filename.c_str();
    this->fp_in.open(fname);

    if (!this->fp_in.is_open()) {
        cout << "Error reading from file - aborting!" << endl;
        throw;
    }
	int count = 0;
    while (this->fp_in.good()) {
        string line;
      
        utilityCore::safeGetline(this->fp_in, line);
        if (!line.empty()) {
            ++count;
            if (count > freq) count = 1;
            if (count == 1){
                vector<string> tokens = utilityCore::tokenizeString(line, sep);
                glm::vec3 pt = glm::vec3(atof(tokens[0].c_str()), atof(tokens[1].c_str()), atof(tokens[2].c_str()));
                points.emplace_back(pt);
            }
        }
    }
	this->numPoints = points.size();
}

std::vector<glm::vec3> PointCloud::getPoints(){
    return this->points;
}

int PointCloud::getNumPoints() {
	return this->numPoints;
}



