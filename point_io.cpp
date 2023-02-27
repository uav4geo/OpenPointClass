#include "point_io.hpp"

void savePointSet(pdal::PointViewPtr pView, const std::string &filename){
    pdal::StageFactory factory;
    std::string driver = pdal::StageFactory::inferWriterDriver(filename);
    if (driver.empty()){
        throw std::runtime_error("Can't infer point cloud writer from " + filename);
    }

    pdal::PointTable table;
    pdal::BufferReader reader;
    reader.addView(pView);
    
    for (auto d : pView->dims()){
        table.layout()->registerOrAssignDim(pView->dimName(d), pView->dimType(d));
    }
    
    pdal::Stage *s = factory.createStage(driver);
    pdal::Options opts;
    opts.add("filename", filename);
    s->setOptions(opts);
    s->setInput(reader);

    s->prepare(table);
    s->execute(table);

    std::cout << "Wrote " << filename << std::endl;
}
