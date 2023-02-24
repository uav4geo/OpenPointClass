#include "scale.hpp"
#include "point_io.hpp"
#include <pdal/filters/VoxelDownsizeFilter.hpp>

Scale::Scale(pdal::PointViewPtr source, double resolution) :
    source(source), resolution(resolution){

    pdal::BufferReader reader;
    reader.addView(source);

    pdal::VoxelDownsizeFilter filter;
    pdal::Options opts;
    opts.add("cell", resolution);
    opts.add("mode", "center");
    filter.setOptions(opts);
    filter.setInput(reader);

    pdal::PointTable *table = new pdal::PointTable();
    filter.prepare(*table);
    auto pvSet = filter.execute(*table);
    pView = *pvSet.begin();

}

void Scale::save(const std::string &filename) const{
    std::cout << "points: " << pView->size() << std::endl;
    savePointSet(pView, filename);
}