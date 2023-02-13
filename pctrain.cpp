#include <iostream>
#include <string>
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
// typedef Point_set::Property_map<int> Imap;
// typedef Point_set::Property_map<uint8_t> UCmap;
typedef Point_set::Property_map<Color> Color_map;

namespace Classification = CGAL::Classification;
typedef Classification::Label_handle Label_handle;
typedef Classification::Feature_handle Feature_handle;
typedef Classification::Label_set Label_set;
typedef Classification::Feature_set Feature_set;
typedef Classification::ETHZ::Random_forest_classifier Classifier;


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

int main(int argc, char **argv){
    cmdLineParse( argc-1 , &argv[1] , params );
    if( !Input.set || !Output.set ) help(argv[0]);

    try {
        // Read points
        std::string filename = Input.value;
        std::string labelDimension = "";

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

        // if (labelDimension.empty()) throw std::runtime_error("Cannot find a classification dimension in the input point cloud (should be either \"Classification\" or \"Label\")");

        std::cout << "Label dimension: " << labelDimension << std::endl;

        pdal::PointTable table;
        pdal::PointViewSet pvSet;

        s->prepare(table);
        pvSet = s->execute(table);
        pdal::PointViewPtr pView = *pvSet.begin();

        if (pView->empty()) {
            throw std::runtime_error("No points could be fetched from cloud");
        }

        // pdal::Dimension::IdList dims = pView->dims();
        Point_set pts;
        Color_map color = pts.add_property_map<Color> ("color").first;

        pts.reserve (pView->size()); 

        pdal::Dimension::Id labelId;
        for (const auto &dim : pView->dims()){
            
        }

        for (pdal::PointId idx = 0; idx < pView->size(); ++idx) {
            auto p = pView->point(idx);
            auto x = p.getFieldAs<double>(pdal::Dimension::Id::X);
            auto y = p.getFieldAs<double>(pdal::Dimension::Id::Y);
            auto z = p.getFieldAs<double>(pdal::Dimension::Id::Z);
            auto it = pts.insert(Point(x, y, z));

            auto r = p.getFieldAs<uint8_t>(pdal::Dimension::Id::Red);
            auto g = p.getFieldAs<uint8_t>(pdal::Dimension::Id::Green);
            auto b = p.getFieldAs<uint8_t>(pdal::Dimension::Id::Blue);
            
            // std::cout << x << " " << y << " " << z << std::endl;
            // exit(0);
            
            Color c = {{ r, g, b }};
            color[*it] = c;
        }

    } catch(pdal::pdal_error& e) {
        std::cerr << "PDAL Error: " << e.what() << std::endl;
        exit(EXIT_FAILURE);
    } catch(std::exception &e){
        std::cerr << "Error: " << e.what() << std::endl;
        exit(EXIT_FAILURE);
    }

    return 0;
}