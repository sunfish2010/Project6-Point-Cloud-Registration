#pragma once

#include <algorithm>
#include <vector>
#include <memory>
#include <glm/glm.hpp>

class Node{
public :
    // can't use ptr for cuda
    //using NodePtr = std::shared_ptr<Node>;
// has to be public due to call in __device__ function
    //NodePtr left;
    //NodePtr right;
    //NodePtr parent;

    int left, right, parent;

    glm::vec3 data;
    int axis;

    Node();
    Node(glm::vec3 &value, int axis);
    Node(const glm::vec3 &value, const NodePtr left, const NodePtr right);
    ~Node()= default;
    //int getAxis();



};

using ptsIter = std::vector<glm::vec3>::iterator;

class KDTree{
public:
    KDTree()= default;
    ~KDTree()= default;
    KDTree(std::vector<glm::vec3> pts);
    std::vector<Node> getTree() const { return tree};

private:
    using ptsIter = std::vector<glm::vec3>::iterator;
    NodePtr make_tree(const ptsIter &begin, const ptsIter &end, int axis);
    std::vector<Node> tree;
};


bool sortX(const glm::vec3 &pt1, glm::vec3& pt2){
    return pt1.x < pt2.x;
}


bool sortY(const glm::vec3 &pt1, glm::vec3& pt2){
    return pt1.y < pt2.y;
}


bool sortZ(const glm::vec3 &pt1, glm::vec3& pt2){
    return pt1.y < pt2.y;
}