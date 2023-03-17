#include "constants.hpp"
#include "point_io.hpp"
#include "randomforest.hpp"

#include "vendor/cxxopts.hpp"

#ifdef WITH_GBM
#include "gbm.hpp"
#endif

int main(int argc, char **argv){
    cxxopts::Options options("pcclassify", "Classifies a point cloud using a precomputed model");
    options.add_options()
            ("i,input", "Input point cloud", cxxopts::value<std::string>())
            ("o,output", "Output point cloud", cxxopts::value<std::string>())
            ("m,model", "Input classification model", cxxopts::value<std::string>()->default_value("model.bin"))
            ("r,regularization", "Regularization method (none, local_smooth)", cxxopts::value<std::string>()->default_value("local_smooth"))
            ("reg-radius", "Regularization radius (meters)", cxxopts::value<double>()->default_value("2.5"))
            ("c,color", "Output a colored point cloud instead of a classified one", cxxopts::value<bool>()->default_value("false"))
            ("u,unclassified", "Only classify points that are labeled as unclassified and leave the others untouched", cxxopts::value<bool>()->default_value("false"))
            ("s,skip", "Do not apply these classification labels (comma separated) and leave them as-is", cxxopts::value<std::vector<int>>())
            ("e,eval", "If the input point cloud is labeled, enable accuracy evaluation", cxxopts::value<bool>()->default_value("false"))
            ("h,help", "Print usage")
        ;
    options.parse_positional({"input", "output", "model"});
    options.positional_help("[input point cloud] [output point cloud] [input classification model]");
    cxxopts::ParseResult result;
    try{
        result = options.parse(argc, argv);
    }catch(const std::exception &e){
        std::cerr << e.what() << std::endl;
        std::cerr << options.help() << std::endl;
        exit(1);
    }

    bool showHelp = false;

    if (result.count("help") || !result.count("input") || !result.count("output")) showHelp = true;
    
    rf::Regularization regularization = rf::Regularization::None;

    try { 
        regularization = rf::parseRegularization(result["regularization"].as<std::string>()); 
    } catch(...) { showHelp = true; }

    if (showHelp){
        std::cout << options.help() << std::endl;
        exit(0);
    }

    try {
        // Read points
        std::string inputFile = result["input"].as<std::string>();
        std::string modelFile = result["model"].as<std::string>();
        std::string outputFile = result["output"].as<std::string>();

        rf::RandomForest *rtrees = rf::loadForest(modelFile);

        auto labels = getTrainingLabels();
        auto pointSet = readPointSet(inputFile);

        double startResolution = rtrees->params.resolution;
        double radius = rtrees->params.radius;
        
        int numScales = rtrees->params.numScales;

        std::cout << "Starting resolution: " << startResolution << std::endl;

        auto features = getFeatures(computeScales(numScales, pointSet, startResolution, radius));
        std::cout << "Features: " << features.size() << std::endl;

        rf::classify(*pointSet, rtrees, features, labels, regularization, 
            result["reg-radius"].as<double>(), result["color"].as<bool>(), result["unclassified"].as<bool>(), result["eval"].as<bool>(), result["skip"].as<std::vector<int>>());
        savePointSet(*pointSet, outputFile);
    } catch(std::exception &e){
        std::cerr << "Error: " << e.what() << std::endl;
        exit(EXIT_FAILURE);
    }

    return 0;
}
