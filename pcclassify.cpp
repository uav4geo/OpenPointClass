#include "common.hpp"
#include "point_io.hpp"
#include "randomforest.hpp"

void help(char *ex){
    std::cout << "Usage: " << ex << std::endl
              << "\t <input point cloud>" << std::endl
              << "\t <classification model (.bin)>" << std::endl
              << "\t <output point cloud>" << std::endl
              << "\t [start resolution]" << std::endl;
    exit(EXIT_FAILURE);
}

int main(int argc, char **argv){
    if( argc < 4 ) help(argv[0]);

    try {
        // Read points
        std::string inputFile = std::string(argv[1]);
        std::string modelFile = std::string(argv[2]);
        std::string outputFile = std::string(argv[3]);

        auto labels = getTrainingLabels();
        auto pointSet = readPointSet(inputFile);

        double startResolution = argc >= 5 ? std::atof(argv[4]) : pointSet.spacing(); // meters
        std::cout << "Starting resolution: " << startResolution << std::endl;
        auto scales = computeScales(NUM_SCALES, pointSet, startResolution);
        std::cout << "Computed " << scales.size() << " scales" << std::endl;

        auto features = getFeatures(scales);
        std::cout << "Features: " << features.size() << std::endl;

        classify(pointSet, modelFile, features, labels, true, false);
        savePointSet(pointSet, outputFile);
    } catch(std::exception &e){
        std::cerr << "Error: " << e.what() << std::endl;
        exit(EXIT_FAILURE);
    }

    return 0;
}
