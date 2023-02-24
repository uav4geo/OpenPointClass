#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/gzip.hpp>

#include "random-forest/node-gini.hpp"
#include "random-forest/forest.hpp"

#include <fstream>
#include <sstream>

int main(int argc, char **argv) {
  // SIFT features
  unsigned int feature_dim = 128;

  // read command-line parameters
  liblearning::RandomForest::ForestParams params;
  params.n_trees   = 25;
  params.max_depth = 20;
  std::string positive_fname, negative_fname, forest_fname, test_fname, result_fname;
  bool verbose = true;

  if (argc == 1) {
    std::cerr << "./rforest -t " << params.n_trees << " -d " << params.max_depth << " -p pos.txt -n neg.txt -f rforest.gz" << std::endl
              << "./rforest -f rforest.gz -i desc.txt -o res.txt" << std::endl
              << "  number of trees:                 -t " << params.n_trees << std::endl
              << "  maximum tree depth:              -d " << params.max_depth << std::endl
              << "  positive training file:          -p [filename]" << std::endl
              << "  negative training file:          -n [filename]" << std::endl
              << "  serialized random forest:        -f [filename]" << std::endl
              << "  descriptors to be classified:    -i [filename]" << std::endl
              << "  classification result:           -o [filename]" << std::endl
              << "  quiet mode:                      -q" << std::endl;
    return(0);
  }

  int arg = 0;
  while (++arg < argc) {
    if (! strcmp(argv[arg], "-t"))
      params.n_trees = atoi(argv[++arg]);
    if (! strcmp(argv[arg], "-d"))
      params.max_depth = atoi(argv[++arg]);
    if (! strcmp(argv[arg], "-p"))
      positive_fname = argv[++arg];
    if (! strcmp(argv[arg], "-n"))
      negative_fname = argv[++arg];
    if (! strcmp(argv[arg], "-f"))
      forest_fname = argv[++arg];
    if (! strcmp(argv[arg], "-i"))
      test_fname = argv[++arg];
    if (! strcmp(argv[arg], "-o"))
      result_fname = argv[++arg];
    if (! strcmp(argv[arg], "-q"))
      verbose = false;
  }

  // empty forest
  liblearning::RandomForest::RandomForest< liblearning::RandomForest::NodeGini<liblearning::RandomForest::AxisAlignedSplitter> > rtrees(params);
  liblearning::RandomForest::AxisAlignedRandomSplitGenerator generator;

  if ((! positive_fname.empty()) && (! negative_fname.empty()) ) {
    std::ifstream str_pos(positive_fname.c_str());
    if (!str_pos.is_open()) {
      std::cerr << "File \"" << positive_fname << "\" does not exist!" << std::endl;
      return(1);
    }

    if (verbose)
      std::cout << "reading pos ..." << std::endl;
    std::istream_iterator<float> in_pos(str_pos);
    std::vector<float> samples_data(in_pos, std::istream_iterator<float>()); // read positive data
    if (verbose)
      std::cout << "  " << samples_data.size()/feature_dim << " samples read" << std::endl;
    std::vector<int> labels_data(samples_data.size()/feature_dim, 1);

    str_pos.close();

    std::ifstream str_neg(negative_fname.c_str());
    if (!str_neg.is_open()) {
      std::cerr << "File \"" << negative_fname << "\" does not exist!" << std::endl;
      return(1);
    }

    if (verbose)
      std::cout << "reading neg ..." << std::endl;
    std::istream_iterator<float> in_neg(str_neg);
    std::copy(in_neg, std::istream_iterator<float>(), back_inserter(samples_data)); // read negative data
    if (verbose)
      std::cout << "  " << samples_data.size()/feature_dim-labels_data.size() << " samples read" << std::endl;
    labels_data.resize(samples_data.size()/feature_dim, 0);

    str_neg.close();

    // convert to DataView2D type
    liblearning::DataView2D<int> labels(&labels_data[0], labels_data.size(), 1);
    liblearning::DataView2D<float> samples(&samples_data[0], samples_data.size()/feature_dim, feature_dim);

    // training - the 0 is a random seed, can be changed to get different random numbers
    if (verbose)
      std::cout << "training " << params.n_trees << " trees with maximum depth " << params.max_depth << std::endl;
    rtrees.train(samples, labels, liblearning::DataView2D<int>(), generator, 0, false);

    // saving the model
    if (verbose)
      std::cout << "saving forest" << std::endl;
    if (! forest_fname.empty()) {
      std::ofstream ofs(forest_fname.c_str(), std::ios_base::out | std::ios_base::binary);
      boost::iostreams::filtering_ostream outs;
      outs.push(boost::iostreams::gzip_compressor());
      outs.push(ofs);
      boost::archive::text_oarchive oas(outs);
      oas << BOOST_SERIALIZATION_NVP(rtrees);
    }
  }

  if (! test_fname.empty()) {

    // loading the model from file
    // do setup part first (actual param values don't matter for loading)
    if (! forest_fname.empty()) {
      std::ifstream ifs(forest_fname.c_str(), std::ios_base::in | std::ios_base::binary);
      if (!ifs.is_open()) {
        std::cerr << "File \"" << forest_fname << "\" does not exist!" << std::endl;
        return(1);
      }
      if (verbose)
        std::cout << "loading forest" << std::endl;
      boost::iostreams::filtering_istream ins;
      ins.push(boost::iostreams::gzip_decompressor());
      ins.push(ifs);
      boost::archive::text_iarchive ias(ins);
      ias >> BOOST_SERIALIZATION_NVP(rtrees);
    }

    // testing
    std::ifstream str_tst(test_fname.c_str());
    if (!str_tst.is_open()) {
      std::cerr << "File \"" << test_fname << "\" does not exist!" << std::endl;
      return(1);
    }

    if (verbose)
      std::cout << "reading tst ..." << std::endl;
    std::istream_iterator<float> in_tst(str_tst);
    // next two don't have to be vectors - what is passed to the tree are just pointers
    std::vector<float> test_data(in_tst, std::istream_iterator<float>()); // read test data
    if (verbose)
      std::cout << "  " << test_data.size()/feature_dim << " samples read" << std::endl;
    std::vector<float> results(rtrees.params.n_classes); // allocate space for results

    str_tst.close();

    if (verbose)
      std::cout << "testing ..." << std::endl;
    std::ofstream str_res;
    if (! result_fname.empty()) {
      str_res.open(result_fname.c_str());
    }
    
    // result = most likely class
    // results = values with probabilities
    unsigned int sum = 0;
    // classify all test data
    for (unsigned int i = 0; i < test_data.size(); i += rtrees.params.n_features) {
      sum += rtrees.evaluate(&test_data[i], &results[0]);
      if (str_res.is_open()) {
        std::ostream_iterator<float> out_res(str_res, " ");
        std::copy(results.begin(), results.end(), out_res); // write space-separated values
        str_res << std::endl; // newline
      }
    }
    
    if (str_res.is_open())
      str_res.close();

    if (verbose)
      std::cout << "  " << sum << " positive samples" << std::endl;
  }
}
