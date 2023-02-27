#ifndef RANDOMFOREST_H
#define RANDOMFOREST_H

#include "random-forest/node-gini.hpp"
#include "random-forest/forest.hpp"
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/gzip.hpp>

#include <pdal/PointView.hpp>
#include <pdal/PointRef.hpp>
#include "features.hpp"
#include "labels.hpp"
#include "common.hpp"

void train(const PointSetData &pointSet, 
    const std::vector<Feature *> &features, 
    const std::vector<Label> &labels,
    const std::string &modelFilename);

void classify(const PointSetData &pointSet, 
    const std::string &modelFilename,
    const std::vector<Feature *> &features, 
    const std::vector<Label> &labels,
    bool useColors = false,
    bool evaluate = false);

#endif