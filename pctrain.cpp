#include "common.hpp"
#include "point_io.hpp"
#include "randomforest.hpp"

#ifdef WITH_GBM
#include "gbm.hpp"
#endif

void help(char *ex){
    std::cout << "Usage: " << ex << std::endl
              << "\t <input labeled point cloud> [...]" << std::endl
              << "\t <output forest model>" << std::endl;
    exit(EXIT_FAILURE);
}

int main(int argc, char **argv){
    if( argc < 3 ) help(argv[0]);

    try {
        std::vector<PointSet *> pointSets;
        std::vector<std::vector<Feature *> > featureSets;
        double startResolution = -1.0;
        std::string filename;

        for (size_t i = 1; i < argc - 1; i++){
            filename = std::string(argv[i]);

            auto pointSet = readPointSet(filename);
            if (startResolution == -1.0){
                startResolution = pointSet->spacing(); // meters
                std::cout << "Starting resolution: " << startResolution << std::endl;
            }

            auto features = getFeatures(computeScales(NUM_SCALES, pointSet, startResolution));
            std::cout << "Features: " << features.size() << std::endl;

            pointSets.push_back(pointSet);
            featureSets.push_back(features);
        }

        auto labels = getTrainingLabels();
        std::string modelFilename = std::string(argv[argc - 1]);

        rf::train(pointSets, featureSets, labels, modelFilename);
        // gbm::train(pointSet, features, labels, modelFilename);

        const std::string ext = filename.substr(filename.length() - 4);
        const std::string evalFilename = filename.substr(0, filename.length() - 4) + "_eval" + ext;
        if (fileExists(evalFilename)){
            std::cout << "Evaluating on " << evalFilename << " ..." << std::endl;
            
            auto evalPointSet = readPointSet(evalFilename);
            auto evalFeatures = getFeatures(computeScales(NUM_SCALES, evalPointSet, startResolution));
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
