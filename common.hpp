#include <iostream>
#include <string>
#include <unordered_map>
#include <random>

#include <json.hpp>

#include <pdal/Options.hpp>
#include <pdal/PointTable.hpp>
#include <pdal/StageFactory.hpp>
#include <pdal/KDIndex.hpp>
#include <pdal/io/BufferReader.hpp>

#include "scale.hpp"
#include "features.hpp"
#include "labels.hpp"

using json = nlohmann::json;

#define NUM_SCALES 2

bool fileExists(const std::string &path){
    std::ifstream fin(path);
    bool e = fin.good();
    fin.close();
    return e;
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

std::pair<pdal::PointViewPtr, pdal::Dimension::Id> readPointSet(const std::string &filename){
    std::string labelDimension = "";

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
    
    pdal::PointTable *table = new pdal::PointTable();
    pdal::PointViewSet pvSet;

    std::cout << "Reading points from " << filename << std::endl;

    s->prepare(*table);
    pvSet = s->execute(*table);

    pdal::PointViewPtr pView = *pvSet.begin();
    if (pView->empty()) {
        throw std::runtime_error("No points could be fetched");
    }

    std::cout << "Number of points: " << pView->size() << std::endl;

    for (auto &d : pView->dims()){
        std::string dim = pView->dimName(d);
        if (dim == "Label" || dim == "label" ||
            dim == "Classification" || dim == "classification" ||
            dim == "Class" || dim == "class"){
            labelDimension = dim;
        }
    }
    
    pdal::PointLayoutPtr layout(table->layout());
    pdal::Dimension::Id labelId;
    if (!labelDimension.empty()){
        std::cout << "Label dimension: " << labelDimension << std::endl;
        labelId = layout->findDim(labelDimension);
    }

    auto trainingCodes = getTrainingCodes();

    // Re-map labels if needed
    if (hasMappings){
        for (pdal::PointId idx = 0; idx < pView->size(); ++idx) {
            auto p = pView->point(idx);
            int label = p.getFieldAs<int>(labelId);

            if (mappings.find(label) != mappings.end()){
                label = trainingCodes[mappings[label]];
            }else{
                label = trainingCodes["unassigned"];
            }

            // p.setField(labelId, label);
            pView->setField(labelId, idx, label);
        }
    }

    return std::make_pair(pView, labelId);
}

double modeSpacing(pdal::PointViewPtr pView, int kNeighbors){
    pdal::KD3Index index(*pView);
    index.build();
    
    pdal::point_count_t np = pView->size();
    pdal::point_count_t SAMPLES = std::min<pdal::point_count_t>(np, 10000);

    pdal::point_count_t count = kNeighbors;
    pdal::PointIdList indices(count);
    std::vector<double> sqr_dists(count);

    std::unordered_map<uint64_t, size_t> dist_map;
    std::vector<double> all_distances;

    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint64_t> randomDis(
        std::numeric_limits<pdal::PointId>::min(),
        np - 1
    );

    #pragma omp parallel private (indices, sqr_dists)
    {
        indices.resize(count);
        sqr_dists.resize(count);

        #pragma omp for
        for (pdal::PointId i = 0; i < SAMPLES; ++i)
        {
            const pdal::PointId idx = randomDis(gen);
            index.knnSearch(idx, count, &indices, &sqr_dists);

            double sum = 0.0;
            for (size_t j = 1; j < count; ++j){
                sum += std::sqrt(sqr_dists[j]);
            }
            sum /= count;

            #pragma omp critical
            {
                uint64_t k = std::ceil(sum * 100);
                if (dist_map.find(k) == dist_map.end()){
                    dist_map[k] = 1;
                }else{
                    dist_map[k] += 1;
                }
            }

            indices.clear(); indices.resize(count);
            sqr_dists.clear(); sqr_dists.resize(count);
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

    return static_cast<double>(d) / 100.0;
}


std::vector<Scale *> computeScales(size_t numScales, pdal::PointViewPtr pView, double startResolution){
    std::vector<Scale *> scales(numScales, nullptr);
    double r = startResolution;

    for (size_t i = 0; i < numScales; i++){
        scales[i] = new Scale(i, pView, r);
        r *= 2.0;
    }

    return scales;
}

