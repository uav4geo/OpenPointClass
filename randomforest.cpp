#include "randomforest.hpp"

namespace rf{

RandomForest *train(const std::vector<std::string> filenames,
          double *startResolution,
          int numScales,
          int numTrees,
          int treeDepth,
          double radius,
          int maxSamples){
  ForestParams params;
  params.n_trees   = numTrees;
  params.max_depth = treeDepth;
  RandomForest* rtrees = new RandomForest(params);
  AxisAlignedRandomSplitGenerator generator;

  std::vector<float> ft;
  std::vector<int> gt;

  getTrainingData(filenames, startResolution, numScales, radius, maxSamples,
    [&ft, &gt](std::vector<Feature *> &features, size_t idx, int g){
      for (std::size_t f = 0; f < features.size(); f++){
          ft.push_back(features[f]->getValue(idx));
      }
      gt.push_back(g);
    },
    [](size_t numFeatures, int numClasses){});
  std::cout << "Using " << gt.size() << " inliers" << std::endl;
    
  LabelDataView label_vector (&(gt[0]), gt.size(), 1);
  FeatureDataView feature_vector(&(ft[0]), gt.size(), ft.size() / gt.size());

  std::cout << "Training..." << std::endl;
  rtrees->train(feature_vector, label_vector, LabelDataView(), generator, 0, false, false);

  rtrees->params.resolution = *startResolution;
  rtrees->params.radius = radius;
  rtrees->params.numScales = numScales;

  return rtrees;
}

void saveForest(RandomForest *rtrees, const std::string &modelFilename){
  std::ofstream ofs(modelFilename.c_str(), std::ios_base::out | std::ios_base::binary | std::ios_base::trunc);
  rtrees->write(ofs);

  std::cout << "Saved " << modelFilename << std::endl;
}

RandomForest *loadForest(const std::string &modelFilename){
  RandomForest *rtrees = new RandomForest();

  std::cout << "Loading " << modelFilename << std::endl;
  std::ifstream ifs(modelFilename.c_str(), std::ios_base::in | std::ios_base::binary);
  if (!ifs.is_open()) throw std::runtime_error("Cannot open " + modelFilename);
  rtrees->read(ifs);

  return rtrees;
}

void classify(PointSet &pointSet, 
    RandomForest *rtrees,
    const std::vector<Feature *> &features, 
    const std::vector<Label> &labels,
    Regularization regularization,
    double regRadius,
    bool useColors,
    bool unclassifiedOnly,
    bool evaluate,
    const std::vector<int> &skip){
  classifyData<float>(pointSet, 
      [&rtrees](float *ft, float *probs){
          rtrees->evaluate (ft, probs);
      },
      features, labels, regularization, regRadius, useColors, unclassifiedOnly, evaluate, skip);
}

}
