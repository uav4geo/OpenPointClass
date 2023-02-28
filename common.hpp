#ifndef COMMON_H
#define COMMON_H

#include <iostream>
#include <string>
#include <unordered_map>
#include <random>

#include "scale.hpp"
#include "features.hpp"
#include "labels.hpp"
#include "point_io.hpp"


#define NUM_SCALES 2

double modeSpacing(const PointSet &pSet, int kNeighbors);
std::vector<Scale *> computeScales(size_t numScales, PointSet pSet, double startResolution);

#endif