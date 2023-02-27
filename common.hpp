#ifndef COMMON_H
#define COMMON_H

#include <iostream>
#include <string>
#include <unordered_map>
#include <random>

#include <json.hpp>

#include <pdal/Options.hpp>
#include <pdal/PointTable.hpp>
#include <pdal/StageFactory.hpp>
#include <pdal/KDIndex.hpp>
#include <pdal/io/BufferReader.hpp>

#include "scale.hpp"
#include "features.hpp"
#include "labels.hpp"

using json = nlohmann::json;

#define NUM_SCALES 2

typedef std::pair<pdal::PointViewPtr, pdal::Dimension::Id> PointSetData;

bool fileExists(const std::string &path);
std::unordered_map<int, std::string> getClassMappings(const std::string &filename);
PointSetData readPointSet(const std::string &filename);
double modeSpacing(pdal::PointViewPtr pView, int kNeighbors);
std::vector<Scale *> computeScales(size_t numScales, pdal::PointViewPtr pView, double startResolution);

#endif