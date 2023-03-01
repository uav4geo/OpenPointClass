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
    std::vector<Eigen::Matrix3d> eigenVectors;
    std::vector<Eigen::Matrix2d> orderAxis;
    std::vector<float> heightMin;
    std::vector<float> heightMax;

    Eigen::Matrix3d computeCovariance(const std::vector<size_t> &neighborIds, const Eigen::Vector3d &medoid);
    Eigen::Vector3d computeMedoid(const std::vector<size_t> &neighborIds);
    Eigen::Vector3d computeCentroid(const std::vector<size_t> &pointIds);
    void computeScaledSet();
    void save(const std::string &filename) const;

    Scale(size_t id, const PointSet &pSet, double resolution, int kNeighbors = 10);
};

#endif