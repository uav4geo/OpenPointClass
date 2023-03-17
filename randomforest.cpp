#include "randomforest.hpp"

namespace rf{

Regularization parseRegularization(const std::string &regularization){
    if (regularization == "none") return None;
    else if (regularization == "local_smooth") return LocalSmooth;
    else throw std::runtime_error("Invalid regularization value: " + regularization);
}

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

  auto labels = getTrainingLabels();
  std::vector<int> gt;
  std::vector<float> ft;

  for (size_t i = 0; i < filenames.size(); i++){
    std::cout << "Processing " << filenames[i] << std::endl;
    auto pointSet = readPointSet(filenames[i]);
    if (!pointSet->hasLabels()){
      std::cout << filenames[i] << " has no labels, skipping..." << std::endl;
      continue;
    }

    if (*startResolution == -1.0){
        *startResolution = pointSet->spacing(); // meters
        std::cout << "Starting resolution: " << *startResolution << std::endl;
    }

    auto scales = computeScales(numScales, pointSet, *startResolution, radius);
    auto features = getFeatures(scales);
    std::cout << "Features: " << features.size() << std::endl;

    std::vector<std::size_t> count (labels.size(), 0);
    std::vector<bool> sampled (pointSet->count(), false);
    std::vector<std::pair<size_t, int> > idxes;

    for (size_t i = 0; i < pointSet->count(); i++){
      int g = pointSet->labels[i];
      if (g != LABEL_UNASSIGNED) {
          size_t idx = pointSet->pointMap[i];
          if (!sampled[idx]){
            idxes.push_back(std::make_pair(idx, g));
            count[std::size_t(g)]++;
            sampled[idx] = true;
          }
      }
    }

    size_t totalSamples = 0;
    std::vector<size_t> samplesPerLabel(count.size(), 0);
    for (size_t i = 0; i < count.size(); i++) totalSamples += count[i];
    if (totalSamples == 0) return rtrees;

    // Allocate samples based on their relative distribution
    for (size_t i = 0; i < samplesPerLabel.size(); i++){
      float perc = static_cast<float>(count[i]) / static_cast<float>(totalSamples);
      samplesPerLabel[i] = std::min<size_t>(
                                std::ceil<size_t>(
                                  perc * static_cast<float>(std::min<size_t>(maxSamples, totalSamples))
                                ), 
                                count[i]);
    }

    std::vector<std::size_t> added (labels.size(), 0);
    std::random_shuffle ( idxes.begin(), idxes.end() );

    for (const auto &p : idxes){
      size_t idx = p.first;
      int g = p.second;
      if (added[std::size_t(g)] < samplesPerLabel[std::size_t(g)]){
        for (std::size_t f = 0; f < features.size(); f++){
          ft.push_back(features[f]->getValue(idx));
        }
        gt.push_back(g);
        added[std::size_t(g)]++;
      }
    }

    for (std::size_t i = 0; i < labels.size(); i++)
      std::cout << " * " << labels[i].getName() << ": " << added[i] << " / " << count[i] << std::endl;
    
    // Free up memory for next
    for (size_t i = 0; i < scales.size(); i++) delete scales[i];
    for (size_t i = 0; i < features.size(); i++) delete features[i];
    RELEASE_POINTSET(pointSet);
  }

  LabelDataView label_vector (&(gt[0]), gt.size(), 1);
  FeatureDataView feature_vector(&(ft[0]), gt.size(), ft.size() / gt.size());

  std::cout << "Using " << gt.size() << " inliers" << std::endl;
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
    bool useColors,
    bool unclassifiedOnly,
    bool evaluate){

  
  std::cout << "Classifying..." << std::endl;
  pointSet.base->labels.resize(pointSet.base->count());

  if (regularization == Regularization::None){
    #pragma omp parallel
    {
    std::vector<float> probs(labels.size(), 0.);
    std::vector<float> ft (features.size());

    #pragma omp for
    for (size_t i = 0; i < pointSet.base->count(); i++ ){
      for (std::size_t f = 0; f < features.size(); f++){
        ft[f] = features[f]->getValue(i);
      }

      rtrees->evaluate (ft.data(), probs.data());

      // Find highest probability
      int bestClass = 0;
      float bestClassVal = 0.f;
    
      // Start from 1 (skip ignored class)
      for (std::size_t j = 1; j < probs.size(); j++){
        if (probs[j] > bestClassVal){
          bestClass = j;
          bestClassVal = probs[j];
        }
      }

      pointSet.base->labels[i] = bestClass;
    }
    } // end pragma omp

  }else if (regularization == Regularization::LocalSmooth){
    std::vector<std::vector<float> > values (labels.size(), std::vector<float> (pointSet.base->count(), -1.));
    
    #pragma omp parallel
    {

    std::vector<float> probs(labels.size(), 0.);
    std::vector<float> ft (features.size());

    #pragma omp for
    for (size_t i = 0; i < pointSet.base->count(); i++){
      for (std::size_t f = 0; f < features.size(); f++){
        ft[f] = features[f]->getValue(i);
      }

      rtrees->evaluate (ft.data(), probs.data());

      for(std::size_t j = 0; j < labels.size(); j++){
        values[j][i] = probs[j];
      }
    }

    }

    std::cout << "Local smoothing..." << std::endl;

    #pragma omp parallel
    {

    std::vector<nanoflann::ResultItem<size_t, float>> radiusMatches;
    std::vector<float> mean (values.size(), 0.);
    auto index = pointSet.base->getIndex<KdTree>();

    const double radius = features[0]->getScale()->radius;

    #pragma omp for schedule(dynamic, 1)
    for (size_t i = 0; i < pointSet.base->count(); i++){
      size_t numMatches = index->radiusSearch(&pointSet.base->points[i][0], radius, radiusMatches);
      std::fill(mean.begin(), mean.end(), 0.);

      for (size_t n = 0; n < numMatches; n++){
        for (std::size_t j = 0; j < values.size(); ++ j){
          mean[j] += values[j][radiusMatches[n].first];
        }
      }

      int bestClass = 0;
      float bestClassVal = 0.f;
      for(std::size_t j = 1; j < mean.size(); j++){
        mean[j] /= numMatches;
        if(mean[j] > bestClassVal){
          bestClassVal = mean[j];
          bestClass = j;
        }
      }

      pointSet.base->labels[i] = bestClass;
    }

    }
  }else{
    throw std::runtime_error("Invalid regularization");
  }

  std::size_t correct = 0;
  if (!useColors && !pointSet.hasLabels()) pointSet.labels.resize(pointSet.count());

  #pragma omp parallel for
  for (size_t i = 0; i < pointSet.count(); i++){
    size_t idx = pointSet.pointMap[i];

    int bestClass = pointSet.base->labels[idx];
    auto label = labels[bestClass];

    if (evaluate){
      if (pointSet.labels[i] == bestClass){
        #pragma omp atomic
        correct++;
      }
    }

    // if unclassifiedOnly, do not update points with an existing classification
    if (unclassifiedOnly && pointSet.hasLabels() 
        && pointSet.labels[i] != LABEL_UNCLASSIFIED) continue;

    // Update point info
    if (useColors){
      auto color = label.getColor();
      pointSet.colors[i][0] = color.r;
      pointSet.colors[i][1] = color.g;
      pointSet.colors[i][2] = color.b;
    }else{
      pointSet.labels[i] = label.getAsprsCode();
    }
  }

  if (evaluate){
    float modelErr = (1.f - static_cast<float>(correct) / static_cast<float>(pointSet.count()));
    std::cout << "Model error: " << std::setprecision(4) << (modelErr * 100.f) << "%" << std::endl;
  }
}

}
