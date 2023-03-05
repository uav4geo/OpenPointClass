#include "constants.hpp"
#include "point_io.hpp"
#include "randomforest.hpp"

#include "vendor/cxxopts.hpp"

#ifdef WITH_GBM
#include "gbm.hpp"
#endif


int main(int argc, char **argv){
    cxxopts::Options options("pctrain", "Trains a point cloud classification model");
    options.add_options()
            ("i,input", "Ground truth labeled point cloud(s) to use for training", cxxopts::value<std::vector<std::string>>())
            ("o,output", "Output model", cxxopts::value<std::string>()->default_value("model.bin"))
            ("r,resolution", "Resolution of the first scale (-1 = estimate automatically)", cxxopts::value<double>()->default_value("-1"))
            ("s,scales", "Number of scales to compute", cxxopts::value<int>()->default_value(MKSTR(NUM_SCALES)))
            ("t,trees", "Number of trees to populate for each input point cloud", cxxopts::value<int>()->default_value(MKSTR(N_TREES)))
            ("d,depth", "Maximum depth of trees", cxxopts::value<int>()->default_value(MKSTR(MAX_DEPTH)))
            ("radius", "Radius size to use for neighbor search (meters)", cxxopts::value<double>()->default_value(MKSTR(RADIUS)))
            ("h,help", "Print usage")
        ;
    options.parse_positional({"input", "output"});
    auto result = options.parse(argc, argv);
    
    if (result.count("help") || !result.count("input")){
        std::cout << options.help() << std::endl;
        exit(0);
    }

    try {
        auto filenames = result["input"].as<std::vector<std::string>>();
        std::string modelFilename = result["output"].as<std::string>();
        
        double startResolution = result["resolution"].as<double>();
        int scales = result["scales"].as<int>();
        int numTrees = result["trees"].as<int>();
        int treeDepth = result["depth"].as<int>();
        double radius = result["radius"].as<double>();

        rf::train(filenames, &startResolution, scales, numTrees, treeDepth, radius, modelFilename);
        // gbm::train(pointSet, features, labels, modelFilename);

        const std::string lastFile = filenames.back();
        const std::string ext = lastFile.substr(lastFile.length() - 4);
        const std::string evalFilename = lastFile.substr(0, lastFile.length() - 4) + "_eval" + ext;
        if (fileExists(evalFilename)){
            std::cout << "Evaluating on " << evalFilename << " ..." << std::endl;
            
            auto labels = getTrainingLabels();
            auto evalPointSet = readPointSet(evalFilename);
            auto evalFeatures = getFeatures(computeScales(scales, evalPointSet, startResolution, radius));
            std::cout << "Features: " << evalFeatures.size() << std::endl;

            rf::classify(*evalPointSet, modelFilename, evalFeatures, labels, rf::Regularization::None, true, true);
            // gbm::classify(evalPointSet, modelFilename, evalFeatures, labels, true, true);
            savePointSet(*evalPointSet, "evaluation.ply");
        }
    } catch(std::exception &e){
        std::cerr << "Error: " << e.what() << std::endl;
        exit(EXIT_FAILURE);
    }

    return 0;
}
