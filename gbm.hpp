#ifndef GBM_H
#define GBM_H

#include "features.hpp"
#include "labels.hpp"
#include "common.hpp"
#include "point_io.hpp"

namespace gbm{
void test();

void train(const PointSet &pointSet, 
    const std::vector<Feature *> &features, 
    const std::vector<Label> &labels,
    const std::string &modelFilename);

void classify(PointSet &pointSet, 
    const std::string &modelFilename,
    const std::vector<Feature *> &features, 
    const std::vector<Label> &labels,
    bool useColors = false,
    bool evaluate = false);

}


#endif