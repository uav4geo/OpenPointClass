#ifndef POINTIO_H
#define POINTIO_H

#include <pdal/Options.hpp>
#include <pdal/PointTable.hpp>
#include <pdal/StageFactory.hpp>
#include <pdal/io/BufferReader.hpp>

void savePointSet(pdal::PointViewPtr pView, const std::string &filename);

#endif