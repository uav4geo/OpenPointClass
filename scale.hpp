#ifndef SCALE_H
#define SCALE_H

#include <pdal/Options.hpp>
#include <pdal/PointTable.hpp>
#include <pdal/StageFactory.hpp>
#include <pdal/KDIndex.hpp>
#include <Eigen/Dense>

class Scale{
    pdal::PointViewPtr pView;
    double resolution;
    int kNeighbors;

    pdal::PointViewPtr pScaledView;
    pdal::KD3Index *kdTree;

    std::vector<Eigen::Vector3d > eigenValues;

    Eigen::Matrix3d computeCovariance(const pdal::PointRef &p, pdal::PointViewPtr neighborView, const pdal::PointIdList &neighborIds);
    Eigen::Vector3d computeMedoid(const pdal::PointRef &p, pdal::PointViewPtr neighborView, const pdal::PointIdList &neighborIds);
public:
    Scale(pdal::PointViewPtr pView, double resolution, int kNeighbors = 10);

    void save(const std::string &filename) const;

    double getResolution() const { return resolution; }
    pdal::PointViewPtr getScaledView() const { return pScaledView; }
};

#endif