#include "features.hpp"

std::vector<Feature *> getFeatures(const std::vector<Scale *> &scales){
    std::vector<Feature *> feats;

    // Multi-scale features
    for (size_t i = 0; i < scales.size(); i++){
        feats.push_back(reinterpret_cast<Feature *>(new SurfaceVariation(scales[i])));
    }

    // Color
    for (size_t c = 0; c < 3; c++){
        feats.push_back(reinterpret_cast<Feature *>(new PointColor(scales[0], c)));
    }
    

    return feats;
}