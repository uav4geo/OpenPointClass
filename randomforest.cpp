#include "randomforest.hpp"

namespace rf{

void train(const PointSet &pointSet, 
          const std::vector<Feature *> &features, 
          const std::vector<Label> &labels, 
          const std::string &modelFilename){
  liblearning::RandomForest::ForestParams params;
  params.n_trees   = N_TREES;
  params.max_depth = MAX_DEPTH;

  liblearning::RandomForest::RandomForest< liblearning::RandomForest::NodeGini<liblearning::RandomForest::AxisAlignedSplitter> > rtrees(params);
  liblearning::RandomForest::AxisAlignedRandomSplitGenerator generator;
  
  std::vector<int> gt;
  std::vector<float> ft;
  std::vector<std::size_t> count (labels.size(), 0);
  std::vector<bool> sampled (pointSet.count(), false);
  std::vector<std::pair<size_t, int> > idxes;

  for (size_t i = 0; i < pointSet.count(); i++){
    int g = pointSet.labels[i];
    if (g != LABEL_UNASSIGNED) {
        size_t idx = pointSet.pointMap[i];
        if (!sampled[idx]){
          idxes.push_back(std::make_pair(idx, g));
          count[std::size_t(g)]++;
          sampled[idx] = true;
        }
    }
  }

  size_t samplesPerLabel = std::numeric_limits<size_t>::max();
  for (std::size_t i = 0; i < labels.size(); i++){
    if (count[i] > 0) samplesPerLabel = std::min(count[i], samplesPerLabel);
  }
  samplesPerLabel = std::min<size_t>(samplesPerLabel, 1000000);
  std::vector<std::size_t> added (labels.size(), 0);

  std::cout << "Samples per label: " << samplesPerLabel << std::endl;

  std::random_shuffle ( idxes.begin(), idxes.end() );

  for (const auto &p : idxes){
    size_t idx = p.first;
    int g = p.second;
    if (added[std::size_t(g)] < samplesPerLabel){
      for (std::size_t f = 0; f < features.size(); f++){
        ft.push_back(features[f]->getValue(idx));
      }
      gt.push_back(g);
      added[std::size_t(g)]++;
    }
  }
  
  std::cout << "Using " << gt.size() << " inliers:" << std::endl;
  for (std::size_t i = 0; i < labels.size(); i++)
      std::cout << " * " << labels[i].getName() << ": " << added[i] << " / " << count[i] << std::endl;

  liblearning::DataView2D<int> label_vector (&(gt[0]), gt.size(), 1);
  liblearning::DataView2D<float> feature_vector(&(ft[0]), gt.size(), ft.size() / gt.size());

  std::cout << "Training..." << std::endl;
  rtrees.train(feature_vector, label_vector, liblearning::DataView2D<int>(), generator, 0, false);
  
  std::ofstream ofs(modelFilename.c_str(), std::ios_base::out | std::ios_base::binary | std::ios_base::trunc);
  boost::iostreams::filtering_ostream outs;
  outs.push(boost::iostreams::gzip_compressor());
  outs.push(ofs);
  boost::archive::text_oarchive oas(outs);
  oas << BOOST_SERIALIZATION_NVP(rtrees);

  std::cout << "Saved " << modelFilename << std::endl;
}

void classify(PointSet &pointSet, 
    const std::string &modelFilename,
    const std::vector<Feature *> &features, 
    const std::vector<Label> &labels,
    Regularization regularization,
    bool useColors,
    bool evaluate){
  std::cout << "Loading " << modelFilename << std::endl;
  std::ifstream ifs(modelFilename.c_str(), std::ios_base::in | std::ios_base::binary);
  if (!ifs.is_open()) throw std::runtime_error("Cannot open " + modelFilename);
  
  liblearning::RandomForest::ForestParams params;
  params.n_trees   = N_TREES;
  params.max_depth = MAX_DEPTH;

  liblearning::RandomForest::RandomForest< liblearning::RandomForest::NodeGini<liblearning::RandomForest::AxisAlignedSplitter> > rtrees(params);
  
  boost::iostreams::filtering_istream ins;
  ins.push(boost::iostreams::gzip_decompressor());
  ins.push(ifs);
  boost::archive::text_iarchive ias(ins);
  ias >> BOOST_SERIALIZATION_NVP(rtrees);

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

      rtrees.evaluate (ft.data(), probs.data());

      // Find highest probability
      int bestClass = 0;
      float bestClassVal = 0.f;
    
      for (std::size_t j = 0; j < probs.size(); j++){
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

      rtrees.evaluate (ft.data(), probs.data());

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

    #pragma omp for
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
      for(std::size_t j = 0; j < mean.size(); j++){
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

  #pragma omp parallel for
  for (size_t i = 0; i < pointSet.count(); i++){
    size_t idx = pointSet.pointMap[i];

    int bestClass = pointSet.base->labels[idx];
    auto label = labels[bestClass];

    if (useColors){
      auto color = label.getColor();
      pointSet.colors[i][0] = color.r;
      pointSet.colors[i][1] = color.g;
      pointSet.colors[i][2] = color.b;
    }else{
      // TODO
    }

    if (evaluate){
      if (pointSet.labels[i] == bestClass){
        #pragma omp atomic
        correct++;
      }
    }
  }

  if (evaluate){
    float modelErr = (1.f - static_cast<float>(correct) / static_cast<float>(pointSet.count()));
    std::cout << "Model error: " << std::setprecision(4) << (modelErr * 100.f) << "%" << std::endl;
  }
}

}