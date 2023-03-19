#ifndef GBM_H
#define GBM_H
#include <LightGBM/config.h>
#include <LightGBM/dataset_loader.h>
#include <LightGBM/boosting.h>
#include <LightGBM/objective_function.h>
#include <LightGBM/metric.h>
#include <LightGBM/utils/common.h>
#include <LightGBM/prediction_early_stop.h>

#include "features.hpp"
#include "labels.hpp"
#include "constants.hpp"
#include "point_io.hpp"

namespace gbm{

typedef LightGBM::Boosting Boosting;

Boosting* train(const std::vector<std::string> filenames,
    double *startResolution,
    int numScales,
    int numTrees,
    int treeDepth,
    double radius,
    int maxSamples);

Boosting *loadBooster(const std::string &modelFilename);
void saveBooster(Boosting *booster, const std::string &modelFilename);

void classify(PointSet &pointSet, 
    Boosting *booster,
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