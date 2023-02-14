#include <iostream>
#include <string>
#include <unordered_map>

#include <json.hpp>

#include <pdal/Options.hpp>
#include <pdal/PointTable.hpp>
#include <pdal/StageFactory.hpp>

#include <CGAL/Simple_cartesian.h>
#include <CGAL/Classification.h>
#include <CGAL/Point_set_3.h>

typedef CGAL::Simple_cartesian<double> Kernel;
typedef Kernel::Point_3 Point;
typedef CGAL::Point_set_3<Point> Point_set;
typedef std::array<uint8_t, 3> Color;

typedef Point_set::Point_map Pmap;
typedef Point_set::Property_map<int> Imap;
typedef Point_set::Property_map<uint8_t> UCmap;
typedef Point_set::Property_map<Color> Color_map;

namespace Classification = CGAL::Classification;
typedef Classification::Label_handle Label_handle;
typedef Classification::Feature_handle Feature_handle;
typedef Classification::Label_set Label_set;
typedef Classification::Feature_set Feature_set;
typedef Classification::ETHZ::Random_forest_classifier Classifier;
typedef Classification::Point_set_feature_generator<Kernel, Point_set, Pmap>    Feature_generator;

using json = nlohmann::json;

static std::unordered_map<std::string, int> trainingCodes = {
    {"unassigned", 0},
    {"ground", 0},
    {"low_vegetation", 1},
    {"medium_vegetation", 2},
    {"high_vegetation", 3},
    {"building", 4},
    {"water", 5},
    {"road_surface", 6},
};

static std::unordered_map<std::string, int> asprsCodes = {
    {"unassigned", 2},
    {"ground", 2},
    {"low_vegetation", 3},
    {"medium_vegetation", 4},
    {"high_vegetation", 5},
    {"building", 6},
    {"noise", 7},
    {"reserved", 8},
    {"water", 9},
    {"rail", 10},
    {"road_surface", 11},
    {"reserved_2", 12},
    {"wire_guard", 13},
    {"wire_conductor", 14},
    {"transmission_tower", 15},
    {"wire_connect", 16},
    {"bridge_deck", 17},
    {"high_noise", 18}
};

std::unordered_map<int, std::string> getClassMappings(const std::string &filename){
    std::string jsonFile = filename.substr(0, filename.length() - 4) + ".json";
    std::ifstream fin(jsonFile);
    std::unordered_map<int, std::string> res;

    if (fin.good()){
        std::cout << "Reading classification information: " << jsonFile << std::endl;
        json data = json::parse(fin);
        if (data.contains("classification")){
            auto c = data["classification"];
            for (json::iterator it = c.begin(); it != c.end(); ++it) {
                res[std::stoi(it.key())] = it.value().get<std::string>();
                std::cout << it.key() << ": " << it.value().get<std::string>() << std::endl;
            }
        }else{
            std::cout << "Error: Invalid JSON (no mapping will be applied)" << std::endl;
        }
    }

    return res;
}

std::unique_ptr<Point_set> readPointSet(const std::string &filename, bool training, Imap *pLabelMap = nullptr){
    std::string labelDimension = "";

    auto mappings = training ? getClassMappings(filename) : std::unordered_map<int, std::string>();
    bool hasMappings = !mappings.empty();

    pdal::StageFactory factory;
    std::string driver = pdal::StageFactory::inferReaderDriver(filename);
    if (driver.empty()){
        throw std::runtime_error("Can't infer point cloud reader from " + filename);
    }

    pdal::Stage *s = factory.createStage(driver);
    pdal::Options opts;
    opts.add("filename", filename);
    s->setOptions(opts);
    
    pdal::PointTable table;
    pdal::PointViewSet pvSet;

    std::cout << "Reading points from " << filename << std::endl;

    s->prepare(table);
    pvSet = s->execute(table);
    pdal::PointViewPtr pView = *pvSet.begin();
    if (pView->empty()) {
        throw std::runtime_error("No points could be fetched");
    }

    std::cout << "Number of points: " << pView->size() << std::endl;

    if (training){
        for (auto &d : pView->dims()){
            std::string dim = pView->dimName(d);
            if (dim == "Label" || dim == "label" ||
                dim == "Classification" || dim == "classification" ||
                dim == "Class" || dim == "class"){
                labelDimension = dim;
            }
        }
        
        if (labelDimension.empty()) throw std::runtime_error("Cannot find a classification dimension in the input point cloud (should be either \"Classification\" or \"Label\")");
        std::cout << "Label dimension: " << labelDimension << std::endl;
    }

    std::unique_ptr<Point_set> pts = std::make_unique<Point_set>();
    pts->reserve (pView->size());

    Color_map colorMap = pts->add_property_map<Color> ("color").first;

    if (training){
       *pLabelMap = pts->add_property_map<int>("label").first;
    }

    pdal::PointLayoutPtr layout(table.layout());
    pdal::Dimension::Id labelId;
    if (training){
        labelId = layout->findDim(labelDimension);
    }

    for (pdal::PointId idx = 0; idx < pView->size(); ++idx) {
        auto p = pView->point(idx);
        auto x = p.getFieldAs<double>(pdal::Dimension::Id::X);
        auto y = p.getFieldAs<double>(pdal::Dimension::Id::Y);
        auto z = p.getFieldAs<double>(pdal::Dimension::Id::Z);
        auto it = pts->insert(Point(x, y, z));

        auto r = p.getFieldAs<uint8_t>(pdal::Dimension::Id::Red);
        auto g = p.getFieldAs<uint8_t>(pdal::Dimension::Id::Green);
        auto b = p.getFieldAs<uint8_t>(pdal::Dimension::Id::Blue);
        
        Color c = {{ r, g, b }};
        colorMap[*it] = c;

        if (training){
            int label = p.getFieldAs<int>(labelId);

            if (hasMappings){
                if (mappings.find(label) != mappings.end()){
                    label = trainingCodes[mappings[label]];
                }else{
                    label = trainingCodes["unassigned"];
                }
            }

            (*pLabelMap)[*it] = label;
        }
    }

    return pts;
}

std::unique_ptr<Feature_generator> getGenerator(const Point_set &pts, int numScales = 9, float resolution = -1.0f){
    std::unique_ptr<Feature_generator> generator = std::make_unique<Feature_generator>(pts, pts.point_map(), numScales, resolution);
    return generator;
}

std::unique_ptr<Feature_set> getFeatures(Feature_generator &generator){
    std::cout << "Generating features..." << std::endl;

    std::unique_ptr<Feature_set> features = std::make_unique<Feature_set>();

    features->begin_parallel_additions();

    // TODO: add your custom features here
    generator.generate_point_based_features (*features);

    features->end_parallel_additions();

    return features;
}

std::unique_ptr<Label_set> getLabels(){
    std::unique_ptr<Label_set> labels = std::make_unique<Label_set>();

    labels->add ("ground");
    labels->add ("low_vegetation");
    labels->add ("medium_vegetation");
    labels->add ("high_vegetation");
    labels->add ("building");
    labels->add ("water");
    labels->add ("road_surface");

    return labels;
}
