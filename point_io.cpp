#include "point_io.hpp"
#include "labels.hpp"

std::string getVertexLine(std::ifstream& reader){
    std::string line;

    // Skip comments
    do {
        std::getline(reader, line);

        if (line.find("element") == 0)
            return line;
        else if (line.find("comment") == 0)
            continue;
        else
            throw std::runtime_error("Invalid PLY file");
    } while (true);
}

size_t getVertexCount(const std::string& line) {

    // Split line into tokens
    std::vector<std::string> tokens;

    std::istringstream iss(line);
    std::string token;
    while (std::getline(iss, token, ' '))
        tokens.push_back(token);

    if (tokens.size() != 3)
        throw std::invalid_argument("Invalid PLY file");

    if (tokens[0] != "element" && tokens[1] != "vertex")
        throw std::invalid_argument("Invalid PLY file");

    return std::stoi(tokens[2]);
}

PointSet readPointSet(const std::string& filename){
    std::ifstream reader(filename);
    
    PointSet r;

    if (!reader.is_open())
        throw std::runtime_error("Cannot open file " + filename);

    std::string line;
    std::getline(reader, line);
    if (line != "ply")
        throw std::runtime_error("Invalid PLY file");
    
    std::getline(reader, line);

    // We are reading an ascii ply
    bool ascii = line == "format ascii 1.0";
    
    const auto vertexLine = getVertexLine(reader);
    const auto count = getVertexCount(vertexLine);

    std::cout << "Reading " << count << " points" << std::endl;

    checkHeader(reader, "x");
    checkHeader(reader, "y");
    checkHeader(reader, "z");

    int c = 0;
    bool hasViews = false;
    bool hasNormals = false;
    bool hasColors = false;
    std::string labelDim = "";

    std::getline(reader, line);
    while (line != "end_header"){
        if (hasHeader(line, "nx") || hasHeader(line, "normal_x")) hasNormals = true;
        if (hasHeader(line, "red")) hasColors = true;
        if (hasHeader(line, "views")) hasViews = true;

        if (hasHeader(line, "label")) labelDim = "label";
        if (hasHeader(line, "classification")) labelDim = "classification";
        if (hasHeader(line, "class")) labelDim = "class";
        
        if (c++ > 100) break;
        std::getline(reader, line);
    }

    bool hasLabels = !labelDim.empty();
        
    r.points.resize(count);
    if (hasNormals) r.normals.resize(count);
    if (hasColors) r.colors.resize(count);
    if (hasViews) r.views.resize(count);
    if (hasLabels) r.labels.resize(count);

    // Read points
    if (ascii){
        uint16_t buf;

        for (size_t i = 0; i < count; i++) {
            reader >> r.points[i][0]
                >> r.points[i][1]
                >> r.points[i][2];
            if (hasNormals){
                reader >> r.normals[i][0]
                    >> r.normals[i][1]
                    >> r.normals[i][2];
            }
            if (hasColors){
                reader >> buf;
                r.colors[i][0] = static_cast<uint8_t>(buf);
                reader >> buf;
                r.colors[i][1] = static_cast<uint8_t>(buf);
                reader >> buf;
                r.colors[i][2] = static_cast<uint8_t>(buf);
            }
            if (hasViews){
                reader >> buf;
                r.views[i] = static_cast<uint8_t>(buf);
            }
            if (hasLabels){
                reader >> buf;
                r.labels[i] = static_cast<uint8_t>(buf);
            }
        }
    }else{

        // Read points
        for (size_t i = 0; i < count; i++) {
            reader.read(reinterpret_cast<char*>(&r.points[i][0]), sizeof(float));
            reader.read(reinterpret_cast<char*>(&r.points[i][1]), sizeof(float));
            reader.read(reinterpret_cast<char*>(&r.points[i][2]), sizeof(float));

            if (hasNormals){
                reader.read(reinterpret_cast<char*>(&r.normals[i][0]), sizeof(float));
                reader.read(reinterpret_cast<char*>(&r.normals[i][1]), sizeof(float));
                reader.read(reinterpret_cast<char*>(&r.normals[i][2]), sizeof(float));
            }

            if (hasColors){
                reader.read(reinterpret_cast<char*>(&r.colors[i][0]), sizeof(uint8_t));
                reader.read(reinterpret_cast<char*>(&r.colors[i][1]), sizeof(uint8_t));
                reader.read(reinterpret_cast<char*>(&r.colors[i][2]), sizeof(uint8_t));
            }

            if (hasViews){
                reader.read(reinterpret_cast<char*>(&r.views[i]), sizeof(uint8_t));
            }

            if (hasLabels){
                reader.read(reinterpret_cast<char*>(&r.labels[i]), sizeof(uint8_t));
            }
        }
    }

    // Re-map labels if needed
    if (hasLabels){
        auto mappings = getClassMappings(filename);
        bool hasMappings = !mappings.empty();

        if (hasMappings){
            auto trainingCodes = getTrainingCodes();
            for (size_t idx = 0; idx < count; idx++) {
                int label = r.labels[idx];

                if (mappings.find(label) != mappings.end()){
                    label = trainingCodes[mappings[label]];
                }else{
                    label = trainingCodes["unassigned"];
                }

                r.labels[idx] = label;
            }
        }
    }

    // for (size_t idx = 0; idx < count; idx++) {
    //     std::cout << r.points[idx][0] << " ";
    //     std::cout << r.points[idx][1] << " ";
    //     std::cout << r.points[idx][2] << " ";

    //     std::cout << std::to_string(r.colors[idx][0]) << " ";
    //     std::cout << std::to_string(r.colors[idx][1]) << " ";
    //     std::cout << std::to_string(r.colors[idx][2]) << " ";

    //     std::cout << std::to_string(r.labels[idx]) << " ";
    //     std::cout << std::endl;
        
    //     if (idx > 9) exit(1);
    // }

    reader.close();

    return r;
}

void checkHeader(std::ifstream& reader, const std::string &prop){
    std::string line;
    std::getline(reader, line);
    if (line.substr(line.length() - prop.length(), prop.length()) != prop){
        throw std::runtime_error("Invalid PLY file (expected 'property * " + prop + "', but found '" + line + "')");
    }
}

bool hasHeader(const std::string &line, const std::string &prop){
    return line.substr(0, 8) == "property" && line.substr(line.length() - prop.length(), prop.length()) == prop;
}

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

        // void write(std::ostream& o) {

        //     const auto cnt = this->points.size();

        //     o << "ply" << std::endl;
        //     o << "format binary_little_endian 1.0" << std::endl;
        //     o << "comment Generated by FPCFilter v" << FPCFilter_VERSION_MAJOR << "." << FPCFilter_VERSION_MINOR << std::endl;
        //     o << "element vertex " << cnt << std::endl;

        //     o << "property float x" << std::endl;
        //     o << "property float y" << std::endl;
        //     o << "property float z" << std::endl;

        //     const auto hasNormals = this->hasNormals();

        //     if (hasNormals)
        //     {
        //         o << "property float nx" << std::endl;
        //         o << "property float ny" << std::endl;
        //         o << "property float nz" << std::endl;
        //     }

        //     o << "property uchar red" << std::endl;
        //     o << "property uchar blue" << std::endl;
        //     o << "property uchar green" << std::endl;
        //     o << "property uchar views" << std::endl;

        //     o << "end_header" << std::endl;

        //     if (hasNormals)
        //     {
        //         for (auto n = 0; n < cnt; n++)
        //         {
        //             const auto point = this->points[n];
        //             const auto extra = this->extras[n];

        //             o.write(reinterpret_cast<const char*>(&point.x), sizeof(float));
        //             o.write(reinterpret_cast<const char*>(&point.y), sizeof(float));
        //             o.write(reinterpret_cast<const char*>(&point.z), sizeof(float));

        //             o.write(reinterpret_cast<const char*>(&extra.nx), sizeof(float));
        //             o.write(reinterpret_cast<const char*>(&extra.ny), sizeof(float));
        //             o.write(reinterpret_cast<const char*>(&extra.nz), sizeof(float));

        //             o.write(reinterpret_cast<const char*>(&point.red), sizeof(uint8_t));
        //             o.write(reinterpret_cast<const char*>(&point.blue), sizeof(uint8_t));
        //             o.write(reinterpret_cast<const char*>(&point.green), sizeof(uint8_t));

        //             o.write(reinterpret_cast<const char*>(&point.views), sizeof(uint8_t));

        //         }

        //     } else
        //     {
        //         for (auto n = 0; n < cnt; n++)
        //         {
        //             const auto point = this->points[n];

        //             o.write(reinterpret_cast<const char*>(&point.x), sizeof(float));
        //             o.write(reinterpret_cast<const char*>(&point.y), sizeof(float));
        //             o.write(reinterpret_cast<const char*>(&point.z), sizeof(float));

        //             o.write(reinterpret_cast<const char*>(&point.red), sizeof(uint8_t));
        //             o.write(reinterpret_cast<const char*>(&point.blue), sizeof(uint8_t));
        //             o.write(reinterpret_cast<const char*>(&point.green), sizeof(uint8_t));

        //             o.write(reinterpret_cast<const char*>(&point.views), sizeof(uint8_t));

        //         }
        //     }            

        // }

std::unordered_map<int, std::string> getClassMappings(const std::string &filename){
    std::string jsonFile = filename.substr(0, filename.length() - 4) + ".json";
    
    // Need to drop _eval filename suffix?
    if (filename.substr(filename.length() - 5 - 4, 5) == "_eval" &&
            !fileExists(jsonFile)){
        jsonFile = filename.substr(0, filename.length() - 5 - 4) + ".json";
    }

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

bool fileExists(const std::string &path){
    std::ifstream fin(path);
    bool e = fin.good();
    fin.close();
    return e;
}