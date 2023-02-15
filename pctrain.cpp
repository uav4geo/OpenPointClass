#include "common.hpp"

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
        
        Color_map colorMap;
        Imap labelMap;
        auto pts = readPointSet(filename, true, &labelMap);

        // TODO: -1.0f should be set to ~4xGSD (/3?) of largest input dataset (currently it's auto-computed)
        auto generator = getGenerator(*pts, 9, -1.0f);
        auto features = getFeatures(*generator);

        // Add labels
        auto labels = getLabels();

        // Check if ground truth is valid for this label set
        if (!labels->is_valid_ground_truth (pts->range(labelMap), true))
            throw std::runtime_error("Invalid ground truth labels; check that the training data has all the required labels.");

        std::cout << "Using ETHZ Random Forest Classifier" << std::endl;
        Classification::ETHZ::Random_forest_classifier classifier(*labels, *features);
        std::cout << "Training..." << std::endl;
        classifier.train (pts->range(labelMap));

        std::string outputFile = std::string(argv[2]);
        std::ofstream out(outputFile, std::ios::binary);
        std::cout << "Saving model to " << outputFile << std::endl;
        classifier.save_configuration(out);
        out.close();
    } catch(pdal::pdal_error& e) {
        std::cerr << "PDAL Error: " << e.what() << std::endl;
        exit(EXIT_FAILURE);
    } catch(std::exception &e){
        std::cerr << "Error: " << e.what() << std::endl;
        exit(EXIT_FAILURE);
    }

    return 0;
}
