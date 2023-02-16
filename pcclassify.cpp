#include "common.hpp"

void help(char *ex){
    std::cout << "Usage: " << ex << std::endl
              << "\t <input point cloud>" << std::endl
              << "\t <classification model (.bin)>" << std::endl
              << "\t <output point cloud>" << std::endl;
    exit(EXIT_FAILURE);
}

int main(int argc, char **argv){
    if( argc < 4 ) help(argv[0]);

    try {
        // Read points
        std::string inputFile = std::string(argv[1]);
        std::string modelFile = std::string(argv[2]);
        std::string outputFile = std::string(argv[3]);

        auto pts = readPointSet(inputFile, false);
        auto generator = getGenerator(*pts);
        auto features = getFeatures(*generator);

        // Add labels
        auto labels = getLabels();

        std::cout << "Using ETHZ Random Forest Classifier" << std::endl;
        Classification::ETHZ::Random_forest_classifier classifier(*labels, *features);
        std::cout << "Loading " << modelFile << std::endl;
        std::ifstream fin(modelFile, std::ios::binary);
        classifier.load_configuration(fin);

        // TODO: test invalid model
        std::vector<int> label_indices(pts->size(), -1);

        std::cout << "Classifying... this might take a bit" << std::endl;
        // Classification::classify_with_graphcut<CGAL::Parallel_tag>
        //    (*pts, pts->point_map(), *labels, classifier,
        //     generator->neighborhood().k_neighbor_query(10),
        //     0.2f, 1, label_indices);
        Classification::classify_with_local_smoothing<CGAL::Parallel_if_available_tag>
              (*pts, pts->point_map(), *labels, classifier,
                generator->neighborhood().sphere_neighbor_query(0.6),
                label_indices);

        // TODO: write using PDAL

        UCmap red = pts->add_property_map<unsigned char>("red", 0).first;
        UCmap green = pts->add_property_map<unsigned char>("green", 0).first;
        UCmap blue = pts->add_property_map<unsigned char>("blue", 0).first;
        for (std::size_t i = 0; i < label_indices.size(); ++ i){
            Label_handle label = (*labels)[label_indices[i]];
            const CGAL::IO::Color& color = label->color();
            red[i] = color.red();
            green[i] = color.green();
            blue[i] = color.blue();
        }

        std::ofstream f (outputFile);
        f.precision(18);
        f << *pts;
        f.close();
        std::cout << "Saved " << outputFile << std::endl;

    } catch(pdal::pdal_error& e) {
        std::cerr << "PDAL Error: " << e.what() << std::endl;
        exit(EXIT_FAILURE);
    } catch(std::exception &e){
        std::cerr << "Error: " << e.what() << std::endl;
        exit(EXIT_FAILURE);
    }

    return 0;
}
