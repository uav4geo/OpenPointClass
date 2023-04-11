#include "constants.hpp"
#include "point_io.hpp"
#include "classifier.hpp"
#include "randomforest.hpp"

#include "vendor/cxxopts.hpp"

#ifdef WITH_GBT
#include "gbm.hpp"
#endif


int main(int argc, char **argv) {
    cxxopts::Options options("pctrain", "Trains a point cloud classification model");
    options.add_options()
        ("i,input", "Ground truth labeled point cloud(s) to use for training", cxxopts::value<std::vector<std::string>>())
        ("o,output", "Output model", cxxopts::value<std::string>()->default_value("model.bin"))
        ("r,resolution", "Resolution of the first scale (-1 = estimate automatically)", cxxopts::value<double>()->default_value("-1"))
        ("s,scales", "Number of scales to compute", cxxopts::value<int>()->default_value(MKSTR(NUM_SCALES)))
        ("t,trees", "Number of trees in the forest", cxxopts::value<int>()->default_value(MKSTR(N_TREES)))
        ("d,depth", "Maximum depth of trees", cxxopts::value<int>()->default_value(MKSTR(MAX_DEPTH)))
        ("m,max-samples", "Approximate maximum number of samples for each input point cloud", cxxopts::value<int>()->default_value("100000"))
        ("radius", "Radius size to use for neighbor search (meters)", cxxopts::value<double>()->default_value(MKSTR(RADIUS)))
        ("e,eval", "Labeled point cloud to use for model accuracy evaluation", cxxopts::value<std::string>()->default_value(""))
        ("eval-result", "Write evaluation results cloud to ply file", cxxopts::value<std::string>()->default_value(""))
        ("stats-file", "Write evaluation statistics to json file", cxxopts::value<std::string>()->default_value(""))
        ("c,classifier", "Which classifier type to use (rf = Random Forest, gbt = Gradient Boosted Trees)", cxxopts::value<std::string>()->default_value("rf"))
        ("classes", "Train only these classification classes (comma separated IDs)", cxxopts::value<std::vector<int>>())
        ("h,help", "Print usage")
        ;
    options.parse_positional({ "input" });
    options.positional_help("[labeled point cloud(s)]");
    cxxopts::ParseResult result;
    try {
        result = options.parse(argc, argv);
    }
    catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        std::cerr << options.help() << std::endl;
        return EXIT_FAILURE;
    }

    if (result.count("help") || !result.count("input")) {
        std::cout << options.help() << std::endl;
        return EXIT_SUCCESS;
    }

    try {
        const auto filenames = result["input"].as<std::vector<std::string>>();
        const auto modelFilename = result["output"].as<std::string>();

        double startResolution = result["resolution"].as<double>();
        const auto scales = result["scales"].as<int>();
        const auto numTrees = result["trees"].as<int>();
        const auto treeDepth = result["depth"].as<int>();
        const auto radius = result["radius"].as<double>();
        const auto maxSamples = result["max-samples"].as<int>();
        const auto classifier = result["classifier"].as<std::string>();
        const auto evalResult = result["eval-result"].as<std::string>();
        const auto statsFile = result["stats-file"].as<std::string>();
        const auto evalFilename = result["eval"].as<std::string>();

        std::vector<int> classes = {};
        if (result.count("classes")) classes = result["classes"].as<std::vector<int>>();

        if (classifier != "rf" && classifier != "gbt") {
            std::cout << options.help() << std::endl;
            return EXIT_FAILURE;
        }

        #ifndef WITH_GBT
        if (classifier == "gbt") {
            std::cerr << "Gradient Boosted Trees support has not been built (try building with -DWITH_GBT=ON)" << std::endl;
            return EXIT_FAILURE;
        }
        #endif 

        std::cout << "Using " << (classifier == "rf" ? "Random Forest" : "Gradient Boosted Trees") << std::endl;

        if (classifier == "rf") {
            rf::RandomForest *rtrees = rf::train(filenames, &startResolution, scales, numTrees, treeDepth, radius, maxSamples, classes);
            rf::saveForest(rtrees, modelFilename);
            delete rtrees;
        }

        #ifdef WITH_GBT
        else if (classifier == "gbt") {
            gbm::Boosting *booster = gbm::train(filenames, &startResolution, scales, numTrees, treeDepth, radius, maxSamples, classes);
            gbm::saveBooster(booster, modelFilename);
        }
        #endif

        if (!evalFilename.empty()) {
            
            std::cout << "Evaluating on " << evalFilename << " ..." << std::endl;

            const ClassifierType ctype = fingerprint(modelFilename);

            rf::RandomForest *rtrees = nullptr;
            #ifdef WITH_GBT
            gbm::Boosting *booster = nullptr;
            #endif

            if (ctype == RandomForest) {
                rtrees = rf::loadForest(modelFilename);
            }

            #ifdef WITH_GBT
            else {
                booster = gbm::loadBooster(modelFilename);
            }
            #endif

            const auto labels = getTrainingLabels();
            const auto evalPointSet = readPointSet(evalFilename);

            if (!evalPointSet->hasLabels()) throw std::runtime_error("Evaluation dataset has no labels");
            const auto evalFeatures = getFeatures(computeScales(scales, evalPointSet, startResolution, radius));
            std::cout << "Features: " << evalFeatures.size() << std::endl;

            if (ctype == RandomForest) {
                rf::classify(*evalPointSet, rtrees, evalFeatures, labels, Regularization::None, 2.5, 
                    true, false, true, {}, statsFile);
            }

            #ifdef WITH_GBT
            else {
                gbm::classify(*evalPointSet, booster, evalFeatures, labels, Regularization::None, 2.5, 
                    true, false, true, {}, statsFile);
            }
            #endif

            if (!evalResult.empty()) {
                savePointSet(*evalPointSet, evalResult);
            }

        }
    }
    catch (std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
