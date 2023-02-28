#ifndef SCALE_H
#define SCALE_H

#include <pdal/Options.hpp>
#include <pdal/PointTable.hpp>
#include <pdal/StageFactory.hpp>
#include <pdal/KDIndex.hpp>
#include <Eigen/Dense>
#include "point_io.hpp"

class Scale{
    size_t id;
    PointSet pSet;
    PointSet scaledSet;
    double resolution;
    int kNeighbors;

    KdTree *kdTree;

    pdal::PointViewPtr pScaledView;
    pdal::PointViewPtr pView;
    // pdal::KD3Index *kdTree;

    Eigen::Matrix3d computeCovariance(size_t ptIdx, const std::vector<size_t> &neighborIds);
    Eigen::Vector3d computeMedoid(size_t ptIdx, const std::vector<size_t> &neighborIds);
    Eigen::Vector3d computeCentroid(const std::vector<size_t> &pointIds);

    void computeScaledSet();
public:
    std::vector<Eigen::Vector3d> eigenValues;

    Scale(size_t id, const PointSet &pSet, double resolution, int kNeighbors = 10);

    size_t getId() const { return id; }
    void save(const std::string &filename) const;
    double getResolution() const { return resolution; }
    pdal::PointViewPtr getScaledView() const { return pScaledView; }
    pdal::PointViewPtr getView() const { return pView; }
    PointSet getScaledSet() const { return scaledSet; }
    PointSet getPointSet() const { return pSet; }
    

};

#endif