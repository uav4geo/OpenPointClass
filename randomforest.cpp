#include "randomforest.hpp"

void testForest(){
  
  liblearning::RandomForest::ForestParams params;
  params.n_trees   = 25;
  params.max_depth = 20;

  liblearning::RandomForest::RandomForest< liblearning::RandomForest::NodeGini<liblearning::RandomForest::AxisAlignedSplitter> > rtrees(params);
  liblearning::RandomForest::AxisAlignedRandomSplitGenerator generator;
  
  std::vector<int> ground_truth; // input

  std::vector<int> gt;
  std::vector<float> ft;
//   std::vector<std::size_t> count (m_labels.size(), 0);

  std::size_t i = 0;
  for (const int& g : ground_truth){
      if (g != -1) {
        // TODO: uncomment
        // for (std::size_t f = 0; f < m_features.size(); f++){
        //   ft.push_back(m_features[f]->value(i));
        // }
        gt.push_back(g);
        // count[std::size_t(g)]++;
      }

      i++;
  }

//   std::cout << "Using " << gt.size() << " inliers:" << std::endl;
//   for (std::size_t i = 0; i < m_labels.size(); i++)
//       std::cout << " * " << m_labels[i]->name() << ": " << count[i] << " inlier(s)" << std::endl;

  liblearning::DataView2D<int> label_vector (&(gt[0]), gt.size(), 1);
  liblearning::DataView2D<float> feature_vector(&(ft[0]), gt.size(), ft.size() / gt.size());
  
  exit(1);
  //rtrees.train(feature_vector, label_vector, liblearning::DataView2D<int>(), generator, 0, false);

}