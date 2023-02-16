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
        auto generator = getGenerator(*pts);
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

        // Evaluate?

        const std::string ext = filename.substr(filename.length() - 4);
        const std::string evalFilename = filename.substr(0, filename.length() - 4) + "_eval" + ext;
        if (fileExists(evalFilename)){
            std::cout << "Evaluating on " << evalFilename << " ..." << std::endl;

            Imap evalLabelMap;
            auto evalPts = readPointSet(evalFilename, true, &evalLabelMap);
            auto evalGenerator = getGenerator(*evalPts);
            auto evalLabels = getLabels();

            if (!labels->is_valid_ground_truth (evalPts->range(evalLabelMap), true))
                throw std::runtime_error("Invalid ground truth labels; check that the evaluation data has all the required labels.");

            std::vector<int> label_indices(evalPts->size(), -1);
            Classification::classify_with_local_smoothing<CGAL::Parallel_if_available_tag>
                (*evalPts, evalPts->point_map(), *evalLabels, classifier,
                    evalGenerator->neighborhood().sphere_neighbor_query(0.6),
                    label_indices);

            std::cout << std::endl << "Evaluation results" << std::endl << "=============" << std::endl;
            Classification::Evaluation evaluation (*evalLabels, evalPts->range(evalLabelMap), label_indices);
            for (Label_handle l : *evalLabels){
                std::cout << " * " << l->name() << ": "
                        << evaluation.precision(l) << " ; "
                        << evaluation.recall(l) << " ; "
                        << evaluation.f1_score(l) << " ; "
                        << evaluation.intersection_over_union(l) << std::endl;
            }
            std::cout << "Accuracy = " << evaluation.accuracy() << std::endl
                    << "Mean F1 score = " << evaluation.mean_f1_score() << std::endl
                    << "Mean IoU = " << evaluation.mean_intersection_over_union() << std::endl;
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
