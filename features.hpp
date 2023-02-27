#ifndef FEATURES_H
#define FEATURES_H

#include "scale.hpp"
#include "color.hpp"

class Feature{
protected:
    std::string name;
    Scale *s = nullptr;
public:
    Feature(Scale *s) : s(s){};

    virtual float getValue(pdal::PointId i) = 0;
    std::string getName() const { return name; }
    pdal::PointRef getPoint(pdal::PointId id) const { return s->getView()->point(id); }

    void setName(const std::string &name){
        this->name = name + "_" + std::to_string(s->getId());
    }
};

class SurfaceVariation : Feature{
public:
    SurfaceVariation(Scale *s) : Feature(s){
        this->setName("surface_variation");
    };

    virtual float getValue(pdal::PointId i){
        return s->eigenValues[i][0];
    }
};

class PointColor : Feature{
private:
    size_t componentIdx;
public:
    PointColor(Scale *s, size_t componentIdx) : Feature(s), componentIdx(componentIdx){
        this->setName("point_color_" + std::to_string(componentIdx));
    };

    virtual float getValue(pdal::PointId i){
        pdal::PointRef p = getPoint(i);
        double r = p.getFieldAs<double>(pdal::Dimension::Id::Red);
        double g = p.getFieldAs<double>(pdal::Dimension::Id::Green);
        double b = p.getFieldAs<double>(pdal::Dimension::Id::Blue);
        auto hsv = rgb2hsv(r, g, b);
        return hsv[componentIdx];
    }
};

std::vector<Feature *> getFeatures(const std::vector<Scale *> &scales);

#endif