#ifndef POINTIO_H
#define POINTIO_H

#include <iostream>
#include <fstream>
#include "vendor/json/json.hpp"
#include "vendor/nanoflann/nanoflann.hpp"

using json = nlohmann::json;

struct XYZ{
    float x;
    float y;
    float z;
};

struct PointSet {
    std::vector<std::array<double, 3> > points;
    std::vector<std::array<uint8_t, 3> > colors;
    
    std::vector<std::array<float, 3> > normals;
    std::vector<uint8_t> labels;
    std::vector<uint8_t> views;

    inline size_t count() const { return points.size(); }
    inline size_t kdtree_get_point_count() const { return points.size(); }
    inline double kdtree_get_pt(const size_t idx, const size_t dim) const{
        return points[idx][dim];
    };
    template <class BBOX>
    bool kdtree_get_bbox(BBOX& /* bb */) const
    {
        return false;
    }

    void appendPoint(const PointSet &src, size_t idx){
        points.push_back(src.points[idx]);
        colors.push_back(src.colors[idx]);
    }

    bool hasNormals() const { return normals.size() > 0; }
    bool hasColors() const { return colors.size() > 0; }
    bool hasViews() const { return views.size() > 0; }
    bool hasLabels() const { return labels.size() > 0; }
};

using KdTree = nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<double, PointSet>,
        PointSet, 3 /* dim */, size_t
        >;
#define KDTREE_MAX_LEAF 20

std::string getVertexLine(std::ifstream& reader);
size_t getVertexCount(const std::string& line);
inline void checkHeader(std::ifstream& reader, const std::string &prop);
inline bool hasHeader(const std::string &line, const std::string &prop);

PointSet readPointSet(const std::string& filename);
void savePointSet(const PointSet &pSet, const std::string &filename);

bool fileExists(const std::string &path);
std::unordered_map<int, std::string> getClassMappings(const std::string &filename);


#endif