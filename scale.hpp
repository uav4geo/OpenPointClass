#ifndef SCALE_H
#define SCALE_H

#include <pdal/Options.hpp>
#include <pdal/PointTable.hpp>
#include <pdal/StageFactory.hpp>

class Scale{
    double resolution;
    pdal::PointViewPtr source;

    pdal::PointViewPtr pView;
public:
    Scale(pdal::PointViewPtr source, double resolution);

    void save(const std::string &filename) const;

    pdal::PointViewPtr getView() const { return pView; }
};

#endif