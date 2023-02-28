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

    virtual float getValue(size_t i) = 0;
    std::string getName() const { return name; }

    void setName(const std::string &name){
        this->name = name + "_" + std::to_string(s->id);
    }
};

class SurfaceVariation : Feature{
public:
    SurfaceVariation(Scale *s) : Feature(s){
        this->setName("surface_variation");
    };

    virtual float getValue(size_t i){
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

    virtual float getValue(size_t i){
        double r = s->pSet.colors[i][0];
        double g = s->pSet.colors[i][1];
        double b = s->pSet.colors[i][2];
        auto hsv = rgb2hsv(r, g, b);
        return hsv[componentIdx];
    }
};

std::vector<Feature *> getFeatures(const std::vector<Scale *> &scales);

#endif