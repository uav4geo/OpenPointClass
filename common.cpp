#include <iostream>
#include "common.hpp"

double modeSpacing(PointSet &pSet, int kNeighbors){
    std::cout << "Estimating mode spacing..." << std::endl;

    auto index = pSet.getIndex<KdTree>();

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
            index->knnSearch(&pSet.points[idx][0], count, &indices[0], &sqr_dists[0]);

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

std::vector<Scale *> computeScales(size_t numScales, PointSet pSet, double startResolution){
    std::vector<Scale *> scales(numScales, nullptr);
    double r = startResolution;

    #pragma omp parallel for
    for (size_t i = 0; i < numScales; i++){
        #pragma omp critical
        {
            std::cout << "Computing scale " << i << "..." << std::endl;
        }
        
        scales[i] = new Scale(i, pSet, r);
        r *= 2.0;
    }

    return scales;
}

