#ifndef POINTIO_H
#define POINTIO_H

#include <iostream>
#include <fstream>
#ifdef WITH_PDAL
#include <pdal/Options.hpp>
#include <pdal/PointTable.hpp>
#include <pdal/StageFactory.hpp>
#include <pdal/io/BufferReader.hpp>
#endif

#include "vendor/json/json.hpp"
#include "vendor/nanoflann/nanoflann.hpp"

using json = nlohmann::json;

struct XYZ {
    float x;
    float y;
    float z;
};

class Bbox3
{
    std::array<float, 6> rep{};

public:

    Bbox3() = default;

    Bbox3(const float x_min, const float y_min, const float z_min,
        const float x_max, const float y_max, const float z_max)
    {
        rep[0] = x_min;
        rep[1] = y_min;
        rep[2] = z_min;
        rep[3] = x_max;
        rep[4] = y_max;
        rep[5] = z_max;
    }

    float xmin() const { return rep[0]; }
    float ymin() const { return rep[1]; }

    float zmin() const { return rep[2]; }
    float xmax() const { return rep[3]; }

    float ymax() const { return rep[4]; }
    float zmax() const { return rep[5]; }

    float min(const int i) const { return rep[i]; }
    float max(const int i) const { return rep[i + 3]; }

    bool overlaps(const Bbox3 &b) const
    {
        return xmin() <= b.xmax() && xmax() >= b.xmin() &&
            ymin() <= b.ymax() && ymax() >= b.ymin() &&
            zmin() <= b.zmax() && zmax() >= b.zmin();
    }

    float operator[](const int i) const { return rep[i]; }

    bool operator==(const Bbox3 &b) const
    {
        return rep[0] == b.rep[0] && rep[1] == b.rep[1] && rep[2] == b.rep[2] &&
            rep[3] == b.rep[3] && rep[4] == b.rep[4] && rep[5] == b.rep[5];
    }

    bool contains(const std::array<float, 3> &point) const
    {
        return rep[0] <= point[0] && rep[3] >= point[0] &&
            rep[1] <= point[1] && rep[4] >= point[1] &&
            rep[2] <= point[2] && rep[5] >= point[2];
    }

    bool operator!=(const Bbox3 &b) const
    {
        return !(*this == b);
    }

};

#define KDTREE_MAX_LEAF 10

#define RELEASE_POINTSET(__POINTER) { if (__POINTER != nullptr) { __POINTER->freeIndex<KdTree>(); delete __POINTER; __POINTER = nullptr; } }

struct PointSet {
    std::vector<std::array<float, 3> > points;
    std::vector<std::array<uint8_t, 3> > colors;

    std::vector<std::array<float, 3> > normals;
    std::vector<uint8_t> labels;
    std::vector<uint8_t> views;

    std::vector<size_t> pointMap;
    PointSet *base = nullptr;

    // Get Bbox3 of the point set
    Bbox3 getBbox() const
    {
        float minx = std::numeric_limits<float>::max();
        float maxx = std::numeric_limits<float>::min();

        float miny = std::numeric_limits<float>::max();
        float maxy = std::numeric_limits<float>::min();

        float minz = std::numeric_limits<float>::max();
        float maxz = std::numeric_limits<float>::min();

        for (auto &point : points)
        {
            minx = std::min(minx, point[0]);
            maxx = std::max(maxx, point[0]);
            miny = std::min(miny, point[1]);
            maxy = std::max(maxy, point[1]);
            minz = std::min(minz, point[2]);
            maxz = std::max(maxz, point[2]);
        }

        return { minx, miny, minz, maxx, maxy, maxz };

    }


    void *kdTree = nullptr;

    #ifdef WITH_PDAL
    pdal::PointViewPtr pointView = nullptr;
    #endif

    template <typename T>
    inline T *getIndex() {
        return kdTree != nullptr ? reinterpret_cast<T *>(kdTree) : buildIndex<T>();
    }

    template <typename T>
    inline T *buildIndex() {
        if (kdTree == nullptr) kdTree = static_cast<void *>(new T(3, *this, { KDTREE_MAX_LEAF }));
        return reinterpret_cast<T *>(kdTree);
    }

    inline size_t count() const { return points.size(); }
    inline size_t kdtree_get_point_count() const { return points.size(); }
    inline float kdtree_get_pt(const size_t idx, const size_t dim) const {
        return points[idx][dim];
    };
    template <class BBOX>
    bool kdtree_get_bbox(BBOX & /* bb */) const
    {
        return false;
    }

    void appendPoint(PointSet &src, size_t idx) {
        points.push_back(src.points[idx]);
        colors.push_back(src.colors[idx]);
    }

    void trackPoint(PointSet &src, size_t idx) {
        src.pointMap[idx] = points.size() - 1;
    }

    bool hasNormals() const { return !normals.empty(); }
    bool hasColors() const { return !colors.empty(); }
    bool hasViews() const { return !views.empty(); }
    bool hasLabels() const { return !labels.empty(); }

    double spacing(int kNeighbors = 3);

    template <typename T>
    void freeIndex() {
        if (kdTree != nullptr) {
            T *tree = getIndex<T>();
            delete tree;
            kdTree = nullptr;
        }
    }

    ~PointSet() {
    }
private:
    double m_spacing = -1.0;
};

using KdTree = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<float, PointSet>,
    PointSet, 3, size_t
>;

std::string getVertexLine(std::ifstream &reader);
size_t getVertexCount(const std::string &line);
inline void checkHeader(std::ifstream &reader, const std::string &prop);
inline bool hasHeader(const std::string &line, const std::string &prop);

PointSet *fastPlyReadPointSet(const std::string &filename);
PointSet *pdalReadPointSet(const std::string &filename);
PointSet *readPointSet(const std::string &filename);

void fastPlySavePointSet(PointSet &pSet, const std::string &filename);
void pdalSavePointSet(PointSet &pSet, const std::string &filename);
void savePointSet(PointSet &pSet, const std::string &filename);


bool fileExists(const std::string &path);
std::unordered_map<int, std::string> getClassMappings(const std::string &filename);


#endif
