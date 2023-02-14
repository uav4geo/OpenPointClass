#include <iostream>
#include <string>
#include <unordered_map>

#include <json.hpp>

#include "CmdLineParser.h"
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
// typedef Point_set::Property_map<uint8_t> UCmap;
typedef Point_set::Property_map<Color> Color_map;

namespace Classification = CGAL::Classification;
typedef Classification::Label_handle Label_handle;
typedef Classification::Feature_handle Feature_handle;
typedef Classification::Label_set Label_set;
typedef Classification::Feature_set Feature_set;
typedef Classification::ETHZ::Random_forest_classifier Classifier;
typedef Classification::Point_set_feature_generator<Kernel, Point_set, Pmap>    Feature_generator;

using json = nlohmann::json;

cmdLineParameter< char* >
    Input( "input" ) ,
    Output( "output" );

cmdLineReadable* params[] = {
    &Input , &Output , 
    NULL
};

void help(char *ex){
    std::cout << "Usage: " << ex << std::endl
              << "\t -" << Input.name << " <input labeled point cloud>" << std::endl
              << "\t -" << Output.name << " <output forest model>" << std::endl;
    exit(EXIT_FAILURE);
}

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

int main(int argc, char **argv){
    cmdLineParse( argc-1 , &argv[1] , params );
    if( !Input.set || !Output.set ) help(argv[0]);

    try {
        // Read points
        std::string filename = Input.value;
        std::string labelDimension = "";

        std::unordered_map<std::string, int> trainingCodes = {
            {"unassigned", 0},
            {"ground", 0},
            {"low_vegetation", 1},
            {"medium_vegetation", 2},
            {"high_vegetation", 3},
            {"building", 4},
            {"water", 5},
            {"road_surface", 6},
        };

        std::unordered_map<std::string, int> asprsCodes = {
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
        auto mappings = getClassMappings(filename);
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
        
        pdal::QuickInfo qi = s->preview();
        if (!qi.valid()){
            throw std::runtime_error("Cannot get quick info for point cloud " + filename);
        }

        std::cout << "Number of points: " << qi.m_pointCount << std::endl;

        for (auto &dim : qi.m_dimNames){
            if (dim == "Label" || dim == "label" ||
                dim == "Classification" || dim == "classification" ||
                dim == "Class" || dim == "class"){
                labelDimension = dim;
            }
        }

        if (labelDimension.empty()) throw std::runtime_error("Cannot find a classification dimension in the input point cloud (should be either \"Classification\" or \"Label\")");

        std::cout << "Label dimension: " << labelDimension << std::endl;

        pdal::PointTable table;
        pdal::PointViewSet pvSet;

        std::cout << "Reading points..." << std::endl;
        s->prepare(table);
        pvSet = s->execute(table);
        pdal::PointViewPtr pView = *pvSet.begin();

        if (pView->empty()) {
            throw std::runtime_error("No points could be fetched from cloud");
        }

        Point_set pts;
        pts.reserve (pView->size());
    
        Color_map color_map = pts.add_property_map<Color> ("color").first;
        Imap label_map = pts.add_property_map<int>("label").first;

        pdal::PointLayoutPtr layout(table.layout());
        pdal::Dimension::Id labelId = layout->findDim(labelDimension);

        for (pdal::PointId idx = 0; idx < pView->size(); ++idx) {
            auto p = pView->point(idx);
            auto x = p.getFieldAs<double>(pdal::Dimension::Id::X);
            auto y = p.getFieldAs<double>(pdal::Dimension::Id::Y);
            auto z = p.getFieldAs<double>(pdal::Dimension::Id::Z);
            auto it = pts.insert(Point(x, y, z));

            auto r = p.getFieldAs<uint8_t>(pdal::Dimension::Id::Red);
            auto g = p.getFieldAs<uint8_t>(pdal::Dimension::Id::Green);
            auto b = p.getFieldAs<uint8_t>(pdal::Dimension::Id::Blue);
            
            Color c = {{ r, g, b }};
            color_map[*it] = c;

            int label = p.getFieldAs<int>(labelId);

            if (hasMappings){
                if (mappings.find(label) != mappings.end()){
                    label = trainingCodes[mappings[label]];
                }else{
                    label = trainingCodes["unassigned"];
                }
            }

            label_map[*it] = label;
        }

        std::cout << "Generating features..." << std::endl;

        Feature_set features;

        Feature_generator generator (pts, pts.point_map(),
                               5);  // using 5 scales TODO: change
        features.begin_parallel_additions();

        // TODO: add your custom features here
        generator.generate_point_based_features (features);

        features.end_parallel_additions();

        // Add labels
        Label_set labels;
        Label_handle ground = labels.add ("ground");
        Label_handle low_vegetation = labels.add ("low_vegetation");
        Label_handle medium_vegetation = labels.add ("medium_vegetation");
        Label_handle high_vegetation = labels.add ("high_vegetation");
        Label_handle building = labels.add ("building");
        Label_handle water = labels.add ("water");
        Label_handle road = labels.add ("road");

        std::cout << labels.size() << std::endl;

        // std::vector<int> ground_truth;
        // ground_truth.reserve (pts.size());
        // std::copy (pts.range(label_map).begin(), pts.range(label_map).end(), std::back_inserter (ground_truth));

        // Check if ground truth is valid for this label set
        if (!labels.is_valid_ground_truth (pts.range(label_map), true))
            throw std::runtime_error("Invalid ground truth labels; check that the training data has all the required labels.");

        std::vector<int> label_indices(pts.size(), -1);
        std::cout << "Using ETHZ Random Forest Classifier" << std::endl;
        Classification::ETHZ::Random_forest_classifier classifier (labels, features);
        std::cout << "Training..." << std::endl;
        classifier.train (pts.range(label_map));
        std::cout << "Done" << std::endl;

    } catch(pdal::pdal_error& e) {
        std::cerr << "PDAL Error: " << e.what() << std::endl;
        exit(EXIT_FAILURE);
    } catch(std::exception &e){
        std::cerr << "Error: " << e.what() << std::endl;
        exit(EXIT_FAILURE);
    }

    return 0;
}
