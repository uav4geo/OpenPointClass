#include "classifier.hpp"

#include "graph.hpp"

Regularization parseRegularization(const std::string &regularization) {
    if (regularization == "none") return None;
    if (regularization == "local_smooth") return LocalSmooth;
    if (regularization == "graph_cut") return GraphCut;
    throw std::runtime_error("Invalid regularization value: " + regularization);
}

ClassifierType fingerprint(const std::string &modelFile) {
    std::ifstream ifs(modelFile.c_str(), std::ios::binary);
    if (!ifs.is_open()) throw std::runtime_error("Cannot open " + modelFile);

    // Read first 4 bytes
    char buf[4] = { 0x00, 0x00, 0x00, 0x00 };
    ifs.read(reinterpret_cast<char *>(&buf), sizeof(char) * 4);

    ifs.close();

    return buf[0] == 0x74 && buf[1] == 0x72 && buf[2] == 0x65 && buf[3] == 0x65 ?
        GradientBoostedTrees :
        RandomForest;
}



void alphaExpansionGraphcut(
    const std::vector<std::pair<std::size_t, std::size_t>> &inputGraph,
    const std::vector<float> &edgeCostMap,
    const std::vector<std::vector<double>> &vertexLabelCostMap,
    const std::vector<std::size_t> &vertexLabelMap)
{

    // Work here

}



