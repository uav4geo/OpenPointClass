#ifndef RANDOMFOREST_H
#define RANDOMFOREST_H

#include <ostream>

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

RandomForest *train(const std::vector<std::string> filenames,
    double *startResolution,
    int numScales,
    int numTrees,
    int treeDepth,
    double radius,
    int maxSamples);

RandomForest *loadForest(const std::string &modelFilename);
void saveForest(RandomForest *rtrees, const std::string &modelFilename);

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
