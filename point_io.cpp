#include <random>
#include "point_io.hpp"
#include "labels.hpp"

double PointSet::spacing(int kNeighbors){
    if (m_spacing != -1) return m_spacing;

    auto index = getIndex<KdTree>();

    size_t np = count();
    size_t SAMPLES = std::min<size_t>(np, 10000);
    int count = kNeighbors + 1;

    std::unordered_map<uint64_t, size_t> dist_map;

    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint64_t> randomDis(
        std::numeric_limits<size_t>::min(),
        np - 1
    );

    std::vector<size_t> indices(count);
    std::vector<float> sqr_dists(count);

    for (size_t i = 0; i < SAMPLES; ++i){
        const size_t idx = randomDis(gen);
        index->knnSearch(&points[idx][0], count, &indices[0], &sqr_dists[0]);

        float sum = 0.0;
        for (size_t j = 1; j < kNeighbors; ++j){
            sum += std::sqrt(sqr_dists[j]);
        }
        sum /= static_cast<float>(kNeighbors);

        uint64_t k = std::ceil(sum * 100);

        if (dist_map.find(k) == dist_map.end()){
            dist_map[k] = 1;
        }else{
            dist_map[k] += 1;
        }
    }

    uint64_t max_val = std::numeric_limits<uint64_t>::min();
    int d = 0;
    for (auto it : dist_map){
        if (it.second > max_val){
            d = it.first;
            max_val = it.second;
        }
    }

    m_spacing = std::max(0.01, static_cast<double>(d) / 100.0);
    return m_spacing;
}

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

    size_t redIdx = 0, greenIdx = 1, blueIdx = 2;

    std::getline(reader, line);
    while (line != "end_header"){
        if (hasHeader(line, "nx") || hasHeader(line, "normal_x")) hasNormals = true;
        if (hasHeader(line, "red")){
            hasColors = true;
            redIdx = c;
        }
        if (hasHeader(line, "green")){
            hasColors = true;
            greenIdx = c;
        }
        if (hasHeader(line, "blue")){
            hasColors = true;
            blueIdx = c;
        }
        if (hasHeader(line, "views")) hasViews = true;

        if (hasHeader(line, "label")) labelDim = "label";
        if (hasHeader(line, "classification")) labelDim = "classification";
        if (hasHeader(line, "class")) labelDim = "class";
        
        if (c++ > 100) break;
        std::getline(reader, line);
    }

    size_t colorIdxMin = std::min<size_t>(std::min<size_t>(redIdx, greenIdx), blueIdx);
    redIdx -= colorIdxMin;
    greenIdx -= colorIdxMin;
    blueIdx -= colorIdxMin;
    if (redIdx + greenIdx + blueIdx != 3) throw std::runtime_error("red/green/blue properties need to be contiguous");

    bool hasLabels = !labelDim.empty();
        
    r.points.resize(count);
    if (hasNormals) r.normals.resize(count);
    if (hasColors) r.colors.resize(count);
    if (hasViews) r.views.resize(count);
    if (hasLabels) r.labels.resize(count);

    // if (hasNormals) std::cout << "N";
    // if (hasColors) std::cout << "C";
    // if (hasViews) std::cout << "V";
    // if (hasLabels) std::cout << "L";
    // std::cout << std::endl;

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
                r.colors[i][redIdx] = static_cast<uint8_t>(buf);
                reader >> buf;
                r.colors[i][greenIdx] = static_cast<uint8_t>(buf);
                reader >> buf;
                r.colors[i][blueIdx] = static_cast<uint8_t>(buf);
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
        XYZ p;
        uint8_t color[3];

        for (size_t i = 0; i < count; i++) {
            reader.read(reinterpret_cast<char*>(&r.points[i][0]), sizeof(float) * 3);

            if (hasNormals){
                reader.read(reinterpret_cast<char*>(&r.normals[i][0]), sizeof(float) * 3);
            }

            if (hasColors){
                reader.read(reinterpret_cast<char*>(&color), sizeof(uint8_t) * 3);
                r.colors[i][redIdx] = color[0];
                r.colors[i][greenIdx] = color[1];
                r.colors[i][blueIdx] = color[2];
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

    //     if (hasLabels){
    //         std::cout << std::to_string(r.labels[idx]) << " ";
    //     }
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
    //std::cout << line << " -> " << prop << " : " << line.substr(line.length() - prop.length(), prop.length()) << std::endl;
    return line.substr(0, 8) == "property" && line.substr(line.length() - prop.length(), prop.length()) == prop;
}

void savePointSet(PointSet &pSet, const std::string &filename){
    std::ofstream o(filename, std::ios::binary);

    o << "ply" << std::endl;
    o << "format binary_little_endian 1.0" << std::endl;
    o << "comment Generated by PCClassify" << std::endl;
    o << "element vertex " << pSet.count() << std::endl;
    o << "property float x" << std::endl;
    o << "property float y" << std::endl;
    o << "property float z" << std::endl;

    bool hasNormals = pSet.hasNormals();
    bool hasColors = pSet.hasColors();
    bool hasViews = pSet.hasViews();
    bool hasLabels = pSet.hasLabels();

    if (hasNormals){
        o << "property float nx" << std::endl;
        o << "property float ny" << std::endl;
        o << "property float nz" << std::endl;
    }
    if (hasColors){
        o << "property uchar red" << std::endl;
        o << "property uchar green" << std::endl;
        o << "property uchar blue" << std::endl;
    }
    if (hasViews){
        o << "property uchar views" << std::endl;
    }
    if (hasLabels){
        // TODO: which name for classification label?
        o << "property uchar class" << std::endl;
    }

    o << "end_header" << std::endl;

    XYZ p;
    for (size_t i = 0; i < pSet.count(); i++){
        o.write(reinterpret_cast<const char*>(&pSet.points[i][0]), sizeof(float) * 3);
        if (hasNormals) o.write(reinterpret_cast<const char*>(&pSet.normals[i][0]), sizeof(float) * 3);
        if (hasColors) o.write(reinterpret_cast<const char*>(&pSet.colors[i][0]), sizeof(uint8_t) * 3);
        if (hasViews) o.write(reinterpret_cast<const char*>(&pSet.views[i]), sizeof(uint8_t));
        if (hasLabels) o.write(reinterpret_cast<const char*>(&pSet.labels[i]), sizeof(uint8_t));
    }

    o.close();
    std::cout << "Wrote " << filename << std::endl;
}

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