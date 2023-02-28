#include "randomforest.hpp"

#define N_TREES 25
#define MAX_DEPTH 20


void train(const PointSetData &pointSet, const std::vector<Feature *> &features, const std::vector<Label> &labels, const std::string &modelFilename){
  pdal::PointViewPtr groundTruth = pointSet.first;
  pdal::Dimension::Id labelId = pointSet.second;

  liblearning::RandomForest::ForestParams params;
  params.n_trees   = N_TREES;
  params.max_depth = MAX_DEPTH;

  liblearning::RandomForest::RandomForest< liblearning::RandomForest::NodeGini<liblearning::RandomForest::AxisAlignedSplitter> > rtrees(params);
  liblearning::RandomForest::AxisAlignedRandomSplitGenerator generator;
  
  std::vector<int> gt;
  std::vector<float> ft;
  std::vector<std::size_t> count (labels.size(), 0);

  for (pdal::PointId i = 0; i < groundTruth->size(); i++ ){
    int g = groundTruth->point(i).getFieldAs<int>(labelId);
    if (g != LABEL_UNASSIGNED) {
        for (std::size_t f = 0; f < features.size(); f++){
          ft.push_back(features[f]->getValue(i));
        }
        gt.push_back(g);
        count[std::size_t(g)]++;
    }
  }

  std::cout << "Using " << gt.size() << " inliers:" << std::endl;
  for (std::size_t i = 0; i < labels.size(); i++)
      std::cout << " * " << labels[i].getName() << ": " << count[i] << std::endl;

  liblearning::DataView2D<int> label_vector (&(gt[0]), gt.size(), 1);
  liblearning::DataView2D<float> feature_vector(&(ft[0]), gt.size(), ft.size() / gt.size());

  std::cout << "Training..." << std::endl;
  rtrees.train(feature_vector, label_vector, liblearning::DataView2D<int>(), generator, 0, false);
  
  std::ofstream ofs(modelFilename.c_str(), std::ios_base::out | std::ios_base::binary);
  boost::iostreams::filtering_ostream outs;
  outs.push(boost::iostreams::gzip_compressor());
  outs.push(ofs);
  boost::archive::text_oarchive oas(outs);
  oas << BOOST_SERIALIZATION_NVP(rtrees);

  std::cout << "Saved " << modelFilename << std::endl;
}

void classify(const PointSetData &pointSet, 
    const std::string &modelFilename,
    const std::vector<Feature *> &features, 
    const std::vector<Label> &labels,
    bool useColors,
    bool evaluate){
  pdal::PointViewPtr input = pointSet.first;

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
  std::vector<float> probs(labels.size(), 0.);
  std::vector<float> ft (features.size());

  std::size_t correct = 0;

  for (pdal::PointId i = 0; i < input->size(); i++ ){
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

    auto label = labels[bestClass];
    if (useColors){
      auto color = label.getColor();
      input->setField(pdal::Dimension::Id::Red, i, color.r);
      input->setField(pdal::Dimension::Id::Green, i, color.g);
      input->setField(pdal::Dimension::Id::Blue, i, color.b);
    }else{
      // TODO
    }

    if (evaluate){
      int gt = input->getFieldAs<int>(pointSet.second, i);
      if (gt == bestClass) correct++;
    }

    // TODO: local smoothing?
  }

  if (evaluate){
    float modelErr = (1.f - static_cast<float>(correct) / static_cast<float>(input->size()));
    std::cout << "Model error: " << std::setprecision(4) << (modelErr * 100.f) << "%" << std::endl;
  }
}