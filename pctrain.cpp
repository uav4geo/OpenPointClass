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

        auto pointSet_new = readPointSet(filename);
        double mSpacing_new = modeSpacing(pointSet_new, 3);
        double startResolution_new = mSpacing_new * 4; // meters

        // TODO: classification mapping
        // auto scales = computeScales(NUM_SCALES, pointSet2, startResolution);

        std::cout << "Starting resolution: " << mSpacing_new << std::endl;
        exit(1);

        auto pointSet = readPointSet_old(filename);
        pdal::PointViewPtr pView = pointSet.first;

        double mSpacing = modeSpacing_old(pView, 3);
        double startResolution = mSpacing * 4; // meters
        std::cout << "Starting resolution: " << startResolution << std::endl;

        auto scales = computeScales(NUM_SCALES, pView, startResolution);
        std::cout << "Computed " << scales.size() << " scales" << std::endl;

        auto features = getFeatures(scales);
        std::cout << "Features: " << features.size() << std::endl;

        auto labels = getTrainingLabels();

        train(pointSet, features, labels, modelFilename);

        const std::string ext = filename.substr(filename.length() - 4);
        const std::string evalFilename = filename.substr(0, filename.length() - 4) + "_eval" + ext;
        if (fileExists(evalFilename)){
            std::cout << "Evaluating on " << evalFilename << " ..." << std::endl;
            
            auto evalPointSet = readPointSet_old(evalFilename);
            auto evalScales = computeScales(NUM_SCALES, evalPointSet.first, startResolution);
            std::cout << "Computed " << evalScales.size() << " scales" << std::endl;
            auto evalFeatures = getFeatures(evalScales);
            std::cout << "Features: " << evalFeatures.size() << std::endl;

            classify(evalPointSet, modelFilename, evalFeatures, labels, true, true);
            savePointSet(evalPointSet.first, "evaluation.ply");
        }
    } catch(pdal::pdal_error& e) {
        std::cerr << "PDAL Error: " << e.what() << std::endl;
        exit(EXIT_FAILURE);
    } catch(std::exception &e){
        std::cerr << "Error: " << e.what() << std::endl;
        exit(EXIT_FAILURE);
    }

    return 0;
}
