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
#include "constants.hpp"
#include "point_io.hpp"

namespace rf{

enum Regularization { None, LocalSmooth};

typedef liblearning::RandomForest::RandomForest< liblearning::RandomForest::NodeGini<liblearning::RandomForest::AxisAlignedSplitter> > RandomForest;
typedef liblearning::RandomForest::AxisAlignedRandomSplitGenerator AxisAlignedRandomSplitGenerator;
typedef liblearning::RandomForest::ForestParams ForestParams;
typedef liblearning::DataView2D<int> LabelDataView;
typedef liblearning::DataView2D<float> FeatureDataView;

Regularization parseRegularization(const std::string &regularization);

void train(const std::vector<std::string> filenames,
    double *startResolution,
    int numScales,
    int numTrees,
    int treeDepth,
    double radius,
    int maxSamplesPerLabel,
    const std::string &modelFilename);

RandomForest *loadForest(const std::string &modelFilename);

void trainForest(const PointSet &pointSet,
          const std::vector<Feature *> &features,
          const std::vector<Label> &labels,
          RandomForest *rtrees,
          int maxSamplesPerLabel);

void classify(PointSet &pointSet, 
    RandomForest *rtrees,
    const std::vector<Feature *> &features, 
    const std::vector<Label> &labels,
    Regularization regularization = Regularization::None,
    bool useColors = false,
    bool unclassifiedOnly = false,
    bool evaluate = false);

}
#endif
