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

    std::vector<Eigen::Vector3f> eigenValues;
    std::vector<Eigen::Matrix3f> eigenVectors;
    std::vector<Eigen::Matrix2f> orderAxis;
    std::vector<float> heightMin;
    std::vector<float> heightMax;
    std::vector<std::array<float, 3> > avgHsv;

    Eigen::Matrix3f computeCovariance(const std::vector<size_t> &neighborIds, const Eigen::Vector3f &medoid);
    Eigen::Vector3f computeMedoid(const std::vector<size_t> &neighborIds);
    Eigen::Vector3f computeCentroid(const std::vector<size_t> &pointIds);
    void computeScaledSet();
    void save(const std::string &filename);

    Scale(size_t id, PointSet &pSet, double resolution, int kNeighbors = 10, double radius = 0.6);
};

std::vector<Scale *> computeScales(size_t numScales, PointSet pSet, double startResolution);

#endif