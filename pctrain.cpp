#include "common.hpp"
#include "point_io.hpp"
#include "randomforest.hpp"

void help(char *ex){
    std::cout << "Usage: " << ex << std::endl
              << "\t <input labeled point cloud>" << std::endl
              << "\t <output forest model>" << std::endl;
    exit(EXIT_FAILURE);
}

int main(int argc, char **argv){
    if( argc < 3 ) help(argv[0]);


    try {
        // Read points
        std::string filename = std::string(argv[1]);
        std::string modelFilename = std::string(argv[2]);

        auto pointSet = readPointSet(filename);
        double mSpacing = modeSpacing(pointSet, 3);
        // double mSpacing = 0.25;
        double startResolution = mSpacing * 4; // meters
        std::cout << "Starting resolution: " << mSpacing << std::endl;

        auto scales = computeScales(NUM_SCALES, pointSet, startResolution);

        auto features = getFeatures(scales);
        std::cout << "Features: " << features.size() << std::endl;

        auto labels = getTrainingLabels();

        train(pointSet, features, labels, modelFilename);

        const std::string ext = filename.substr(filename.length() - 4);
        const std::string evalFilename = filename.substr(0, filename.length() - 4) + "_eval" + ext;
        if (fileExists(evalFilename)){
            std::cout << "Evaluating on " << evalFilename << " ..." << std::endl;
            
            auto evalPointSet = readPointSet(evalFilename);
            auto evalScales = computeScales(NUM_SCALES, evalPointSet, startResolution);
            std::cout << "Computed " << evalScales.size() << " scales" << std::endl;
            auto evalFeatures = getFeatures(evalScales);
            std::cout << "Features: " << evalFeatures.size() << std::endl;

            classify(evalPointSet, modelFilename, evalFeatures, labels, true, true);
            savePointSet(evalPointSet, "evaluation.ply");
        }
    } catch(std::exception &e){
        std::cerr << "Error: " << e.what() << std::endl;
        exit(EXIT_FAILURE);
    }

    return 0;
}
