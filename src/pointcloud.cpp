//
// Created by Yu Sun on 11/2/18.
//

#include <iostream>
#include "pointcloud.h"
#include <cstring>
#include <glm/gtx/string_cast.hpp>
using namespace std;

PointCloud::PointCloud(std::string filename) {
    cout << "Reading Point Cloud From " << filename << "..." << endl;
    char* fname = (char*)filename.c_str();
    fp_in.open(fname);

    if (!fp_in.is_open()) {
        cout << "Error reading from file - aborting!" << endl;
        throw;
    }
    while (fp_in.good()) {
        string line;
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty()) {
            vector<string> tokens = utilityCore::tokenizeString(line);
            vec4 pt = glm::vec4(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()),
                                atof(tokens[4].c_str()));
            points.emplace_back(pt);

        }
    }
}

std::vector<glm::vec4> getPoints(){
    return points;
}



