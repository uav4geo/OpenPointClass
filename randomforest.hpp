#ifndef RANDOMFOREST_H
#define RANDOMFOREST_H

#include <ostream>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/gzip.hpp>

#include "random-forest/node-gini.hpp"
#include "random-forest/forest.hpp"

#include "features.hpp"
#include "labels.hpp"
#include "common.hpp"
#include "point_io.hpp"

namespace rf{

enum Regularization { None, LocalSmooth};

void train(const PointSet &pointSet, 
    const std::vector<Feature *> &features, 
    const std::vector<Label> &labels,
    const std::string &modelFilename);

void classify(PointSet &pointSet, 
    const std::string &modelFilename,
    const std::vector<Feature *> &features, 
    const std::vector<Label> &labels,
    Regularization regularization = Regularization::None,
    bool useColors = false,
    bool evaluate = false);

}
#endif