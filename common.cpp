#include <iostream>
#include "common.hpp"

PointSetData readPointSet_old(const std::string &filename){
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

double modeSpacing(const PointSet &pSet, int kNeighbors){
    std::cout << "Estimating mode spacing..." << std::endl;

    KdTree index(3, pSet, { KDTREE_MAX_LEAF });
    index.buildIndex();

    size_t np = pSet.count();
    size_t SAMPLES = std::min<size_t>(np, 10000);
    int count = kNeighbors + 1;

    std::unordered_map<uint64_t, size_t> dist_map;

    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint64_t> randomDis(
        std::numeric_limits<size_t>::min(),
        np - 1
    );

    #pragma omp parallel
    {
        std::vector<size_t> indices(count);
        std::vector<double> sqr_dists(count);

        #pragma omp for
        for (size_t i = 0; i < SAMPLES; ++i){
            const size_t idx = randomDis(gen);
            index.knnSearch(&pSet.points[idx][0], count, &indices[0], &sqr_dists[0]);

            double sum = 0.0;
            for (size_t j = 1; j < kNeighbors; ++j){
                sum += std::sqrt(sqr_dists[j]);
            }
            sum /= static_cast<double>(kNeighbors);

            uint64_t k = std::ceil(sum * 100);

            #pragma omp critical
            {
                if (dist_map.find(k) == dist_map.end()){
                    dist_map[k] = 1;
                }else{
                    dist_map[k] += 1;
                }
            }
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

double modeSpacing_old(pdal::PointViewPtr pView, int kNeighbors){
    pdal::KD3Index index(*pView);
    index.build();
    
    pdal::point_count_t np = pView->size();
    pdal::point_count_t SAMPLES = std::min<pdal::point_count_t>(np, 10000);

    pdal::point_count_t count = kNeighbors;
    pdal::PointIdList indices(count);
    std::vector<double> sqr_dists(count);

    std::unordered_map<uint64_t, size_t> dist_map;

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

std::vector<Scale *> computeScales(size_t numScales, PointSet pSet, double startResolution){
    std::vector<Scale *> scales(numScales, nullptr);
    double r = startResolution;

    //#pragma omp parallel for
    for (size_t i = 0; i < numScales; i++){
        std::cout << "Computing scale " << i << "..." << std::endl;
        scales[i] = new Scale(i, pSet, r);
        r *= 2.0;
    }

    return scales;
}

std::vector<Scale *> computeScales_old(size_t numScales, pdal::PointViewPtr pView, double startResolution){
    // std::vector<Scale *> scales(numScales, nullptr);
    // double r = startResolution;

    // #pragma omp parallel for
    // for (size_t i = 0; i < numScales; i++){
    //     std::cout << "Computing scale " << i << "..." << std::endl;
    //     scales[i] = new Scale(i, pView, r);
    //     r *= 2.0;
    // }

    // return scales;
}
