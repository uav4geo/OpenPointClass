#ifndef POINTIO_H
#define POINTIO_H

#include <pdal/Options.hpp>
#include <pdal/PointTable.hpp>
#include <pdal/StageFactory.hpp>
#include <pdal/io/BufferReader.hpp>

#include "vendor/json/json.hpp"
#include "vendor/nanoflann/nanoflann.hpp"

using json = nlohmann::json;

struct PointSet {
    std::vector<std::array<float, 3> > points;
    std::vector<std::array<uint8_t, 3> > colors;
    
    std::vector<std::array<float, 3> > normals;
    std::vector<uint8_t> labels;
    std::vector<uint8_t> views;

    inline size_t count() const { return points.size(); }
    inline size_t kdtree_get_point_count() const { return points.size(); }
    inline float kdtree_get_pt(const size_t idx, const size_t dim) const{
        return points[idx][dim];
    };
    template <class BBOX>
    bool kdtree_get_bbox(BBOX& /* bb */) const
    {
        return false;
    }
};

using KdTree = nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<float, PointSet>,
        PointSet, 3 /* dim */
        >;
#define KDTREE_MAX_LEAF 20

std::string getVertexLine(std::ifstream& reader);
size_t getVertexCount(const std::string& line);
inline void checkHeader(std::ifstream& reader, const std::string &prop);
inline bool hasHeader(const std::string &line, const std::string &prop);

PointSet readPointSet(const std::string& filename);
void savePointSet(pdal::PointViewPtr pView, const std::string &filename);

bool fileExists(const std::string &path);
std::unordered_map<int, std::string> getClassMappings(const std::string &filename);


#endif