#include "common.hpp"
#include "point_io.hpp"

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
        
        auto pView = readPointSet(filename);
        double mSpacing = modeSpacing(pView, 3);
        double startResolution = mSpacing * 4; // meters

        std::cout << "1st pyramid resolution: " << mSpacing << std::endl;

        savePointSet(pView, "test.ply");
        Scale s(pView, startResolution);
        s.save("1.ply");

        Scale s2(s.getView(), startResolution * 2);
        s2.save("2.ply");

        // for (pdal::PointId idx = 0; idx < pView->size(); ++idx) {
        //     auto p = pView->point(idx);
        //     std::cout << p.getFieldAs<double>(pdal::Dimension::Id::X) << std::endl;
        // }

        // const std::string ext = filename.substr(filename.length() - 4);
        // const std::string evalFilename = filename.substr(0, filename.length() - 4) + "_eval" + ext;
        // if (fileExists(evalFilename)){
        //     std::cout << "Evaluating on " << evalFilename << " ..." << std::endl;

        //     Imap evalLabelMap;
        //     Color_map evalColorMap;
        //     auto evalPts = readPointSet(evalFilename, &evalLabelMap, &evalColorMap);
        //     auto evalGenerator = getGenerator(*evalPts);
        //     auto evalFeatures = getFeatures(*generator, evalColorMap);

        //     if (!labels->is_valid_ground_truth (evalPts->range(evalLabelMap), true))
        //         throw std::runtime_error("Invalid ground truth labels; check that the evaluation data has all the required labels.");

        //     std::vector<int> label_indices(evalPts->size(), -1);

        //     std::cout << "Evaluating..." << std::endl;
        //     Classification::classify_with_local_smoothing<CGAL::Parallel_if_available_tag>
        //         (*evalPts, evalPts->point_map(), *labels, classifier,
        //             evalGenerator->neighborhood().sphere_neighbor_query(0.6),
        //             label_indices);

        //     std::cout << std::endl << "Evaluation results" << std::endl << "==================" << std::endl;
        //     Classification::Evaluation evaluation (*labels, evalPts->range(evalLabelMap), label_indices);
        //     for (Label_handle l : *labels){
        //         std::cout << " * " << l->name() << ": "
        //                 << evaluation.precision(l) << " ; "
        //                 << evaluation.recall(l) << " ; "
        //                 << evaluation.f1_score(l) << " ; "
        //                 << evaluation.intersection_over_union(l) << std::endl;
        //     }
        //     std::cout << "Accuracy = " << evaluation.accuracy() << std::endl
        //             << "Mean F1 score = " << evaluation.mean_f1_score() << std::endl
        //             << "Mean IoU = " << evaluation.mean_intersection_over_union() << std::endl;
        // }


    } catch(pdal::pdal_error& e) {
        std::cerr << "PDAL Error: " << e.what() << std::endl;
        exit(EXIT_FAILURE);
    } catch(std::exception &e){
        std::cerr << "Error: " << e.what() << std::endl;
        exit(EXIT_FAILURE);
    }

    return 0;
}
