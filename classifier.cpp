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
    const std::vector<std::size_t> &vertexLabelMap,
    const size_t numLabels,
    const size_t numVertices)
{

    // Indexes numLabels and numVertices are wrong
    
    std::vector<std::size_t> currentLabels = vertexLabelMap;  // Initialize with input labels

    for (std::size_t alpha = 0; alpha < numLabels; ++alpha)
    {
        Graph graph;

        // Adding nodes to the graph
        std::vector<Graph::node_id> nodes(numVertices);
        for (std::size_t i = 0; i < numVertices; ++i)
        {
            nodes[i] = graph.add_node();

            // Set t-weights. If a vertex's current label is alpha, connect it to SOURCE, otherwise connect it to SINK
            if (currentLabels[i] == alpha)
            {
                graph.set_tweights(nodes[i], vertexLabelCostMap[i][alpha], 0);
            }
            else
            {
                graph.set_tweights(nodes[i], 0, vertexLabelCostMap[i][alpha]);
            }
        }

        // Adding edges to the graph
        for (std::size_t i = 0; i < inputGraph.size(); ++i)
        {
            const std::size_t from = inputGraph[i].first;
            const std::size_t to = inputGraph[i].second;
            graph.add_edge(nodes[from], nodes[to], edgeCostMap[i], edgeCostMap[i]);
        }

        // Compute maxflow
        Graph::captype flow = graph.maxflow();

        // Update labels
        for (std::size_t i = 0; i < numVertices; ++i)
        {
            if (graph.what_segment(nodes[i]) == Graph::SOURCE)
            {
                currentLabels[i] = alpha;
            }
        }
    }

}



