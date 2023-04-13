#include "constants.hpp"
#include "point_io.hpp"
#include "classifier.hpp"
#include "randomforest.hpp"

#include "vendor/cxxopts.hpp"

#ifdef WITH_GBT
#include "gbm.hpp"
#endif

int main(int argc, char **argv) {
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
        ("eval-result", "Write evaluation results cloud to ply file", cxxopts::value<std::string>()->default_value(""))
        ("stats-file", "Write evaluation statistics to json file", cxxopts::value<std::string>()->default_value(""))
        ("h,help", "Print usage")
        ;
    options.parse_positional({ "input", "output", "model" });
    options.positional_help("[input point cloud] [output point cloud] [input classification model]");
    cxxopts::ParseResult result;
    try {
        result = options.parse(argc, argv);
    }
    catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        std::cerr << options.help() << std::endl;
        return EXIT_FAILURE;
    }

    bool showHelp = false;

    if (result.count("help") || !result.count("input") || !result.count("output")) showHelp = true;

    Regularization regularization = Regularization::None;

    try {
        regularization = parseRegularization(result["regularization"].as<std::string>());
    }
    catch (...) { showHelp = true; }

    if (showHelp) {
        std::cout << options.help() << std::endl;
        return 0;
    }

    try {
        // Read points
        const auto inputFile = result["input"].as<std::string>();
        const auto modelFile = result["model"].as<std::string>();
        const auto outputFile = result["output"].as<std::string>();
        std::vector<int> skip = {};
        if (result.count("skip")) skip = result["skip"].as<std::vector<int>>();

        ClassifierType ctype = fingerprint(modelFile);
        #ifndef WITH_GBT
        if (ctype == GradientBoostedTrees) throw std::runtime_error(modelFile + " is a GBT model but GBT support has not been built (try building with -DWITH_GBT=ON)");
        #endif

        std::cout << "Model: " << (ctype == RandomForest ? "Random Forest" : "Gradient Boosted Trees") << std::endl;
        rf::RandomForest *rtrees;
        #ifdef WITH_GBT
        gbm::Boosting *booster;
        #endif

        double startResolution;
        double radius;
        int numScales;

        if (ctype == RandomForest) {
            rtrees = rf::loadForest(modelFile);
            startResolution = rtrees->params.resolution;
            radius = rtrees->params.radius;
            numScales = rtrees->params.numScales;
        }
        #ifdef WITH_GBT
        else {
            booster = gbm::loadBooster(modelFile);
            const gbm::BoosterParams p = gbm::extractBoosterParams(booster);
            startResolution = p.resolution;
            radius = p.radius;
            numScales = p.numScales;
        }
        #endif

        const auto labels = getTrainingLabels();
        const auto pointSet = readPointSet(inputFile);

        std::cout << "Starting resolution: " << startResolution << std::endl;

        const auto features = getFeatures(computeScales(numScales, pointSet, startResolution, radius));
        std::cout << "Features: " << features.size() << std::endl;

        const auto eval = result["eval"].as<bool>();
        const auto evalResult = result["eval-result"].as<std::string>();
        const auto statsFile = result["stats-file"].as<std::string>();
        const auto regRadius = result["reg-radius"].as<double>();
        const auto color = result["color"].as<bool>();
        const auto unclassified = result["unclassified"].as<bool>();

        if (ctype == RandomForest) {
            rf::classify(*pointSet, rtrees, features, labels, regularization,
                regRadius, color, unclassified, eval, skip, statsFile);
        }
        #ifdef WITH_GBT
        else {
            gbm::classify(*pointSet, booster, features, labels, regularization,
                regRadius, color, unclassified, eval, skip, statsFile);
        }
        #endif

        if (eval && !evalResult.empty())
        {
            savePointSet(*pointSet, evalResult);
        }
        
    }
    catch (std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return 0;
}
