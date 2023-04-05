#ifndef RANDOMFOREST_H
#define RANDOMFOREST_H

#include <ostream>

#include "random-forest/node-gini.hpp"
#include "random-forest/forest.hpp"

#include "classifier.hpp"

namespace rf {

typedef liblearning::RandomForest::RandomForest< liblearning::RandomForest::NodeGini<liblearning::RandomForest::AxisAlignedSplitter> > RandomForest;
typedef liblearning::RandomForest::AxisAlignedRandomSplitGenerator AxisAlignedRandomSplitGenerator;
typedef liblearning::RandomForest::ForestParams ForestParams;
typedef liblearning::DataView2D<int> LabelDataView;
typedef liblearning::DataView2D<float> FeatureDataView;


RandomForest *train(const std::vector<std::string> &filenames,
    double *startResolution,
    int numScales,
    int numTrees,
    int treeDepth,
    double radius,
    int maxSamples,
    const std::vector<int> &classes);

RandomForest *loadForest(const std::string &modelFilename);
void saveForest(RandomForest *rtrees, const std::string &modelFilename);

void classify(PointSet &pointSet,
    RandomForest *rtrees,
    const std::vector<Feature *> &features,
    const std::vector<Label> &labels,
    Regularization regularization = Regularization::None,
    double regRadius = 2.5f,
    bool useColors = false,
    bool unclassifiedOnly = false,
    bool evaluate = false,
    const std::vector<int> &skip = {});

}
#endif
