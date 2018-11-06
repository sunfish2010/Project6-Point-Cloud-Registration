#include "kdtree.h"

Node::Node():left(-1), right(-1), parent(-1),
            data(glm::vec3(0.f, 0.f, 0.f), axis(0)){}

Node::Node(const glm::vec3 &value, int axis):left(-1), right(-1), parent(-1), data(value), axis(axis) {}

Node::Node(const glm::vec3 &value, int left,
        int right):left(left), right(right), parent(-1), data(value), axis(0) {}

KDTree::KDTree() {}

KDTree::KDTree(std::vector <glm::vec3> pts) {
    //std::sort(pts.begin(), pts.end(), sortX());
    this->tree = std::vector<Node>(pts.size(), Node());
    make_tree(pts.begin(), pts.end(), 0, pts.size(), tree, 0);

}


void KDTree::make_tree(ptsIter &begin, ptsIter &end, int axis, int length, std::vector<Node>& tree, int index) {
    // just edge case checking, will it happen?
    if (begin == end) return -1;

    if (axis == 0)
        std::sort(begin, end, sortX);
    else if (axis == 1)
        std::sort(begin, end, sortY);
    else
        std::sort(begin, end, sortZ);

    auto mid = begin + (length / 2);

    int llen = length / 2;
    int rlen = length - llen - 1;

    tree[index] = Node(*mid, axis);

    if (llen > 0){
        leftNode = make_tree(begin, mid, (axis + 1) % 3, llen, tree, index + 1);
    }else{
        leftNode = -1;
    }
    if (rlen > 0){
        rightNode = make_tree(mid + 1, end, (axis + 1) % 3, rlen, tree, index + mid + 1);
    }else{
        rightNode = -1;
    }
    tree[index].left = left;
    tree[index].right = right;
    if (left != -1) tree[index].left.parent = index;
    if (right != -1) tree[index].right.parent = index;

}