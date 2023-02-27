#ifndef SCALE_H
#define SCALE_H

#include <pdal/Options.hpp>
#include <pdal/PointTable.hpp>
#include <pdal/StageFactory.hpp>
#include <pdal/KDIndex.hpp>
#include <Eigen/Dense>

class Scale{
    size_t id;
    pdal::PointViewPtr pView;
    double resolution;
    int kNeighbors;

    pdal::PointViewPtr pScaledView;
    pdal::KD3Index *kdTree;

    Eigen::Matrix3d computeCovariance(const pdal::PointRef &p, pdal::PointViewPtr neighborView, const pdal::PointIdList &neighborIds);
    Eigen::Vector3d computeMedoid(const pdal::PointRef &p, pdal::PointViewPtr neighborView, const pdal::PointIdList &neighborIds);
public:
    std::vector<Eigen::Vector3d> eigenValues;

    Scale(size_t id, pdal::PointViewPtr pView, double resolution, int kNeighbors = 10);

    size_t getId() const { return id; }
    void save(const std::string &filename) const;
    double getResolution() const { return resolution; }
    pdal::PointViewPtr getScaledView() const { return pScaledView; }
    pdal::PointViewPtr getView() const { return pView; }
    

};

#endif