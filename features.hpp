#ifndef FEATURES_H
#define FEATURES_H

#include <Eigen/Dense>
#include "scale.hpp"

class Feature{
protected:
    std::string name;
    Scale *s = nullptr;
public:
    Feature(Scale *s) : s(s){};
    virtual ~Feature(){};

    virtual float getValue(size_t i) = 0;
    std::string getName() const { return name; }

    void setName(const std::string &name){
        this->name = name + "_" + std::to_string(s->id);
    }

    Scale *getScale() { return s; }
};

class Omnivariance : public Feature{
public:
    Omnivariance(Scale *s) : Feature(s){
        this->setName("omnivariance");
    };
    virtual float getValue(std::size_t i){
        return std::cbrt(s->eigenValues[i][0] *
                         s->eigenValues[i][1] *
                         s->eigenValues[i][2]);
    }
};

class Eigenentropy : public Feature{
public:
    Eigenentropy(Scale *s) : Feature(s){
        this->setName("eigenentropy");
    };
    virtual float getValue(std::size_t i){
        float sum = 0;
        for (std::size_t j = 0; j < 3; j++){
            float l = s->eigenValues[i][j];
            sum += l * std::log(l);
        }
        return -sum;
    }
};


class Anisotropy : public Feature{
public:
    Anisotropy(Scale *s) : Feature(s){
        this->setName("anisotropy");
    };
    virtual float getValue(std::size_t i){
        return (s->eigenValues[i][2] - 
                s->eigenValues[i][0]) / s->eigenValues[i][2];
    }
};

class Planarity : public Feature{
public:
    Planarity(Scale *s) : Feature(s){
        this->setName("planarity");
    };
    virtual float getValue(std::size_t i){
        return (s->eigenValues[i][1] - 
                s->eigenValues[i][0]) / s->eigenValues[i][2];
    }
};

class Linearity : public Feature{
public:
    Linearity(Scale *s) : Feature(s){
        this->setName("linearity");
    };
    virtual float getValue(std::size_t i){
        return (s->eigenValues[i][2] - 
                s->eigenValues[i][1]) / s->eigenValues[i][2];
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

class Scatter : Feature{
public:
    Scatter(Scale *s) : Feature(s){
        this->setName("scatter");
    };

    virtual float getValue(size_t i){
        return s->eigenValues[i][0] / s->eigenValues[i][2];
    }
};

class Verticality : Feature{
    Eigen::Vector3f up;
public:
    Verticality(Scale *s) : Feature(s), up(0, 0, 1){
        this->setName("verticality");
    };

    virtual float getValue(size_t i){
        return 1.0f - std::fabs(up.dot(s->eigenVectors[i].col(0)));
    }
};

class OrderAxis : Feature{
    size_t order;
    size_t axis;
public:
    OrderAxis(Scale *s, size_t order, size_t axis) 
      : Feature(s), order(order), axis(axis) {
        this->setName("order_" + std::to_string(order) + "_axis_" + std::to_string(axis));
    };

    virtual float getValue(size_t i){
        return s->orderAxis[i](order - 1, axis - 1);
    }
};

class VerticalRange : Feature{
public:
    VerticalRange(Scale *s) : Feature(s) {
        this->setName("vertical_range");
    };

    virtual float getValue(size_t i){
        return s->heightMax[i] - s->heightMin[i];
    }
};

class HeightBelow : Feature{
public:
    HeightBelow(Scale *s) : Feature(s) {
        this->setName("height_below");
    };

    virtual float getValue(size_t i){
        return s->pSet->points[i][2] - s->heightMin[i];
    }
};

class HeightAbove : Feature{
public:
    HeightAbove(Scale *s) : Feature(s) {
        this->setName("height_above");
    };

    virtual float getValue(size_t i){
        return s->heightMax[i] - s->pSet->points[i][2];
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
        double r = s->pSet->colors[i][0];
        double g = s->pSet->colors[i][1];
        double b = s->pSet->colors[i][2];
        auto hsv = rgb2hsv(r, g, b);
        return hsv[componentIdx];
    }
};

class NeighborhoodColors : Feature{
private:
    size_t componentIdx;
public:
    NeighborhoodColors(Scale *s, size_t componentIdx) : Feature(s), componentIdx(componentIdx){
        this->setName("neighborhood_colors_" + std::to_string(componentIdx));
    };

    virtual float getValue(size_t i){
        return s->avgHsv[i][componentIdx];
    }
};

std::vector<Feature *> getFeatures(const std::vector<Scale *> &scales);

#endif