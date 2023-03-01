#ifndef SCALE_H
#define SCALE_H

#include <Eigen/Dense>
#include "point_io.hpp"
#include "color.hpp"

struct Scale{
    size_t id;
    PointSet pSet;
    PointSet scaledSet;
    double resolution;
    int kNeighbors;
    double radius;

    KdTree *kdTree;
    std::vector<Eigen::Vector3d> eigenValues;
    std::vector<Eigen::Matrix3d> eigenVectors;
    std::vector<Eigen::Matrix2d> orderAxis;
    std::vector<float> heightMin;
    std::vector<float> heightMax;
    std::vector<std::array<float, 3> > avgHsv;

    Eigen::Matrix3d computeCovariance(const std::vector<size_t> &neighborIds, const Eigen::Vector3d &medoid);
    Eigen::Vector3d computeMedoid(const std::vector<size_t> &neighborIds);
    Eigen::Vector3d computeCentroid(const std::vector<size_t> &pointIds);
    void computeScaledSet();
    void save(const std::string &filename);

    Scale(size_t id, PointSet &pSet, double resolution, int kNeighbors = 10, double radius = 0.6);
};

#endif