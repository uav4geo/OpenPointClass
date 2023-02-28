#ifndef COMMON_H
#define COMMON_H

#include <iostream>
#include <string>
#include <unordered_map>
#include <random>

#include <pdal/Options.hpp>
#include <pdal/PointTable.hpp>
#include <pdal/StageFactory.hpp>
#include <pdal/KDIndex.hpp>
#include <pdal/io/BufferReader.hpp>

#include "scale.hpp"
#include "features.hpp"
#include "labels.hpp"
#include "point_io.hpp"


#define NUM_SCALES 2

typedef std::pair<pdal::PointViewPtr, pdal::Dimension::Id> PointSetData;

PointSetData readPointSet_old(const std::string &filename);
double modeSpacing_old(pdal::PointViewPtr pView, int kNeighbors);
double modeSpacing(const PointSet &pSet, int kNeighbors);
std::vector<Scale *> computeScales(size_t numScales, pdal::PointViewPtr pView, double startResolution);

#endif