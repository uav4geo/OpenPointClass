#ifndef SCALE_H
#define SCALE_H

#include <Eigen/Dense>
#include "point_io.hpp"

struct Scale{
    size_t id;
    PointSet pSet;
    PointSet scaledSet;
    double resolution;
    int kNeighbors;

    KdTree *kdTree;
    std::vector<Eigen::Vector3d> eigenValues;

    Eigen::Matrix3d computeCovariance(size_t ptIdx, const std::vector<size_t> &neighborIds);
    Eigen::Vector3d computeMedoid(size_t ptIdx, const std::vector<size_t> &neighborIds);
    Eigen::Vector3d computeCentroid(const std::vector<size_t> &pointIds);
    void computeScaledSet();
    void save(const std::string &filename) const;

    Scale(size_t id, const PointSet &pSet, double resolution, int kNeighbors = 10);
};

#endif