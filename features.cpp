#include "features.hpp"

std::vector<Feature *> getFeatures(const std::vector<Scale *> &scales){
    std::vector<Feature *> feats;

    // Multi-scale features
    for (size_t i = 0; i < scales.size(); i++){
        // Covariance
        feats.push_back(reinterpret_cast<Feature *>(new Omnivariance(scales[i])));
        feats.push_back(reinterpret_cast<Feature *>(new Eigenentropy(scales[i])));
        feats.push_back(reinterpret_cast<Feature *>(new Anisotropy(scales[i])));
        feats.push_back(reinterpret_cast<Feature *>(new Planarity(scales[i])));
        feats.push_back(reinterpret_cast<Feature *>(new Linearity(scales[i])));
        feats.push_back(reinterpret_cast<Feature *>(new SurfaceVariation(scales[i])));
        feats.push_back(reinterpret_cast<Feature *>(new Scatter(scales[i])));
        feats.push_back(reinterpret_cast<Feature *>(new Verticality(scales[i])));
        
        // Moments
        feats.push_back(reinterpret_cast<Feature *>(new OrderAxis(scales[i], 1, 1)));
        feats.push_back(reinterpret_cast<Feature *>(new OrderAxis(scales[i], 1, 2)));
        feats.push_back(reinterpret_cast<Feature *>(new OrderAxis(scales[i], 2, 1)));
        feats.push_back(reinterpret_cast<Feature *>(new OrderAxis(scales[i], 2, 2)));

        // Height
        feats.push_back(reinterpret_cast<Feature *>(new VerticalRange(scales[i])));
        feats.push_back(reinterpret_cast<Feature *>(new HeightBelow(scales[i])));
        feats.push_back(reinterpret_cast<Feature *>(new HeightAbove(scales[i])));
    }

    // Color
    for (size_t c = 0; c < 3; c++){
        feats.push_back(reinterpret_cast<Feature *>(new PointColor(scales[0], c)));
        feats.push_back(reinterpret_cast<Feature *>(new NeighborhoodColors(scales[0], c)));
    }

    return feats;
}