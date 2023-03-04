#include <LightGBM/config.h>
#include <LightGBM/dataset_loader.h>
#include <LightGBM/boosting.h>
#include <LightGBM/objective_function.h>
#include <LightGBM/metric.h>
#include <LightGBM/utils/common.h>
#include <LightGBM/prediction_early_stop.h>

#include <iostream>
#include <random>
#include <algorithm>

#include "gbm.hpp"

namespace gbm{

void test(){
    return;
    std::random_device rd;
    std::mt19937 gen(rd());

    // one random generator for every class
    std::vector<std::normal_distribution<>> dists = {
        std::normal_distribution<>(0, 1),
        std::normal_distribution<>(10, 1),
        std::normal_distribution<>(100, 50),
        
        };

    const int numSamples = 5000;
    const int numFeats = 3;
    const int numClasses = static_cast<int>(dists.size());

    std::cout << "Classes: " << numClasses << std::endl;

    std::vector<float> labels(numSamples);
    for (int i=0; i < numSamples; i++){
        labels[i] = i % numClasses;
    }

    std::vector< std::vector<double> > features(numSamples);
    for (int i=0; i < numSamples; i++){
        features[i].resize(numFeats);
        for (int j=0; j < numFeats; j++){
            const auto lbl = static_cast<int>(labels[i]);
            features[i][j] = dists[lbl](gen);
        }
    }

    LightGBM::Config ioconfig;
    ioconfig.num_class = numClasses;
    ioconfig.max_bin = 255;
    ioconfig.verbosity = 10;
    ioconfig.data_random_seed = 1;

    std::vector< std::vector<double> > sampleData(numFeats);
    std::vector< std::vector<int> > sampleIdx(numFeats);

    for (int i=0; i < numSamples; i++){
        for (int j=0; j < numFeats; j++){
            sampleData[j].push_back(features[i][j]);
            sampleIdx[j].push_back(i);
        }
    }

    std::unique_ptr<LightGBM::Dataset> dset;

    LightGBM::DatasetLoader loader(ioconfig, nullptr, numClasses, nullptr);
    dset.reset( loader.ConstructFromSampleData(LightGBM::Common::Vector2Ptr<double>(&sampleData).data(), 
                                               LightGBM::Common::Vector2Ptr<int>(&sampleIdx).data(), 
                                               numFeats,
                                               LightGBM::Common::VectorSize<double>(sampleData).data(),
                                               numSamples,
                                               numSamples,
                                               numSamples) );
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < numSamples; ++i){
        dset->PushOneRow(omp_get_thread_num(), i, features[i]);
    }

    dset->FinishLoad();

    // check bins
    for(int j=0; j < numFeats; j++){
        const auto nbins = dset->FeatureBinMapper(j)->num_bin();
        std::cout << "Feat " << j << std::endl;
        std::cout << "   " << dset->FeatureBinMapper(j)->BinToValue(0) << " ";
        std::cout << "   " << dset->FeatureBinMapper(j)->BinToValue(nbins-2) << " ";
        std::cout << std::endl;
    }
    
    if (!dset->SetFloatField("label", labels.data(), numSamples)) {
        throw std::runtime_error("Error setting label");
    }


    LightGBM::Config boostConfig;
    boostConfig.num_iterations = 5;
    boostConfig.bagging_freq = 1;
    boostConfig.bagging_fraction = 0.5;
    boostConfig.num_class = numClasses;

    // tree params
    boostConfig.min_data_in_leaf = 10;
    boostConfig.num_leaves = 16;

    LightGBM::Config objConfig;
    objConfig.num_class = numClasses;
    // objConfig.label_gain.resize(numClasses, 1.0);

    auto *objFunc = LightGBM::ObjectiveFunction::CreateObjectiveFunction("multiclass", objConfig);
    objFunc->Init(dset->metadata(), dset->num_data());

    LightGBM::Config metricConfig;
    metricConfig.num_class = numClasses;

    std::vector< std::unique_ptr<LightGBM::Metric> > trainMetrics;
    auto metric = std::unique_ptr<LightGBM::Metric>(
        LightGBM::Metric::CreateMetric("multi_logloss", metricConfig));
    metric->Init(dset->metadata(), dset->num_data());
    trainMetrics.push_back(std::move(metric));
    auto *booster = LightGBM::Boosting::CreateBoosting("gbdt", nullptr);

    booster->Init(&boostConfig, dset.get(), objFunc,
        LightGBM::Common::ConstPtrInVectorWrapper<LightGBM::Metric>(trainMetrics));

    // Add if you want to add validation data (eval)
    // booster->AddValidDataset(dset.get(), LightGBM::Common::ConstPtrInVectorWrapper<LightGBM::Metric>(trainMetrics));

    for (int i = 0; i < boostConfig.num_iterations; i++){
        std::cout << "Iteration " << (i+1) << std::endl;

        auto scores = booster->GetEvalAt(0);
        for(auto &v: scores) std::cout << "Score: " << v << std::endl;

        if (booster->TrainOneIter(nullptr, nullptr)){
            std::cout << "Breaking.." << std::endl;
            break;
        }
    }

    std::vector<int> predictedClass(numSamples);
    std::vector<double> predVec(numClasses);

    LightGBM::PredictionEarlyStopConfig early_stop_config;
    auto earlyStop = LightGBM::CreatePredictionEarlyStopInstance("none", early_stop_config);

    booster->InitPredict(booster->GetCurrentIteration() - 1, 0, false);

    for (int i=0; i < numSamples; i++){
        booster->Predict(features[i].data(), predVec.data(), &earlyStop);
        const auto predMax = std::max_element(predVec.begin(), predVec.end());
        predictedClass[i] = std::distance(predVec.begin(), predMax);
    }

    // compute error
    double err = 0;
    for (int i=0; i < numSamples; i++) {
        if (predictedClass[i] != labels[i]){
            err++;
        }
    }
    err /= labels.size();

    std::cout << "Training error: " << err << std::endl;
    exit(1);
}

void train(const PointSet &pointSet, 
    const std::vector<Feature *> &features, 
    const std::vector<Label> &labels,
    const std::string &modelFilename){

    std::vector<float> gt;
    std::vector< std::vector<double> > featureRows;
    std::vector< std::vector<double> > featuresData(features.size());
    std::vector< std::vector<int> > featuresIdx(features.size());
    std::vector<std::size_t> count (labels.size(), 0);
    std::vector<bool> sampled (pointSet.count(), false);
    std::vector<std::pair<size_t, int> > idxes;

    for (size_t i = 0; i < pointSet.count(); i++){
        int g = pointSet.labels[i];
        if (g != LABEL_UNASSIGNED) {
            size_t idx = pointSet.pointMap[i];
            if (!sampled[idx]){
                idxes.push_back(std::make_pair(idx, g));
                count[std::size_t(g)]++;
                sampled[idx] = true;
            }
        }
    }

    size_t samplesPerLabel = std::numeric_limits<size_t>::max();
    for (std::size_t i = 0; i < labels.size(); i++){
        if (count[i] > 0) samplesPerLabel = std::min(count[i], samplesPerLabel);
    }
    samplesPerLabel = std::min<size_t>(samplesPerLabel, 1000000);
    std::vector<std::size_t> added (labels.size(), 0);

    std::cout << "Samples per label: " << samplesPerLabel << std::endl;

    std::random_shuffle ( idxes.begin(), idxes.end() );
    
    int c = 0;
    for (const auto &p : idxes){
        size_t idx = p.first;
        int g = p.second;
        if (added[std::size_t(g)] < samplesPerLabel){
            featureRows.emplace_back();
            featureRows[c].resize(features.size(), 0);
            for (std::size_t f = 0; f < features.size(); f++){
                featureRows[c][f] = features[f]->getValue(idx);
                featuresData[f].push_back(featureRows[c][f]);
                featuresIdx[f].push_back(c);
            }
            gt.push_back(g);
            added[std::size_t(g)]++;
            c++;
        }
    }

    std::cout << "Using " << gt.size() << " inliers:" << std::endl;
    for (std::size_t i = 0; i < labels.size(); i++){
        std::cout << " * " << labels[i].getName() << ": " << added[i] << " / " << count[i] << std::endl;
    }

    const size_t numRows = gt.size();
    const size_t numFeats = features.size();
    int numClasses = labels.size();

    LightGBM::Config ioconfig;
    ioconfig.num_class = numClasses;
    ioconfig.max_bin = 255;

    std::unique_ptr<LightGBM::Dataset> dset;
    LightGBM::DatasetLoader loader(ioconfig, nullptr, numClasses, nullptr);
    dset.reset( loader.ConstructFromSampleData(LightGBM::Common::Vector2Ptr<double>(&featuresData).data(), 
                                               LightGBM::Common::Vector2Ptr<int>(&featuresIdx).data(), 
                                               numFeats,
                                               LightGBM::Common::VectorSize<double>(featuresData).data(),
                                               numRows,
                                               numRows,
                                               numRows) );
    
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < numRows; ++i){
        dset->PushOneRow(omp_get_thread_num(), i, featureRows[i]);
    }

    dset->FinishLoad();

    for(int j = 0; j < numFeats; j++){
        const auto nbins = dset->FeatureBinMapper(j)->num_bin();
        std::cout << features[j]->getName() << std::endl;
        std::cout << "   " << dset->FeatureBinMapper(j)->BinToValue(0) << " ";
        std::cout << "   " << dset->FeatureBinMapper(j)->BinToValue(nbins-2) << " ";
        std::cout << std::endl;
    }
    
    if (!dset->SetFloatField("label", gt.data(), numRows)) {
        throw std::runtime_error("Error setting label");
    }

    LightGBM::Config boostConfig;
    boostConfig.num_iterations = N_TREES;
    // boostConfig.bagging_freq = 1;
    boostConfig.bagging_fraction = 0.5;
    boostConfig.num_class = numClasses;
    boostConfig.max_depth = MAX_DEPTH;

    // tree params
    boostConfig.num_leaves = 16;
    boostConfig.learning_rate = 0.2;

    LightGBM::Config objConfig;
    objConfig.num_class = numClasses;

    auto *objFunc = LightGBM::ObjectiveFunction::CreateObjectiveFunction("multiclass", objConfig);
    objFunc->Init(dset->metadata(), dset->num_data());

    LightGBM::Config metricConfig;
    metricConfig.num_class = numClasses;

    std::vector< std::unique_ptr<LightGBM::Metric> > trainMetrics;
    auto metric = std::unique_ptr<LightGBM::Metric>(
        LightGBM::Metric::CreateMetric("multi_logloss", metricConfig));
    metric->Init(dset->metadata(), dset->num_data());
    trainMetrics.push_back(std::move(metric));
    auto *booster = LightGBM::Boosting::CreateBoosting("gbdt", nullptr);

    booster->Init(&boostConfig, dset.get(), objFunc,
        LightGBM::Common::ConstPtrInVectorWrapper<LightGBM::Metric>(trainMetrics));

    // Add if you want to add validation data (eval)
    // booster->AddValidDataset(dset.get(), LightGBM::Common::ConstPtrInVectorWrapper<LightGBM::Metric>(trainMetrics));

    for (int i = 0; i < boostConfig.num_iterations; i++){
        std::cout << "Iteration " << (i+1) << std::endl;

        auto scores = booster->GetEvalAt(0);
        for(auto &v: scores) std::cout << "Score: " << v << std::endl;

        if (booster->TrainOneIter(nullptr, nullptr)){
            std::cout << "Breaking.." << std::endl;
            break;
        }
    }

    booster->SaveModelToFile(0, 0, 0, modelFilename.c_str());
    std::cout << "Saved " << modelFilename << std::endl;
}

void classify(PointSet &pointSet, 
    const std::string &modelFilename,
    const std::vector<Feature *> &features, 
    const std::vector<Label> &labels,
    bool useColors,
    bool evaluate){
    std::cout << "Loading " << modelFilename << std::endl;

    auto *booster = LightGBM::Boosting::CreateBoosting("gbdt", nullptr);
    if (!LightGBM::Boosting::LoadFileToBoosting(booster, modelFilename.c_str())){
        throw std::runtime_error("Cannot open " + modelFilename);
    }
    booster->InitPredict(0, 0, false);

    LightGBM::PredictionEarlyStopConfig early_stop_config;
    auto earlyStop = LightGBM::CreatePredictionEarlyStopInstance("none", early_stop_config);

    std::cout << "Classifying..." << std::endl;

    std::vector<double> probs(labels.size(), 0.);
    std::vector<double> ft (features.size());
    size_t correct = 0;

    for (size_t i = 0; i < pointSet.count(); i++ ){
        size_t idx = pointSet.pointMap[i];
        for (std::size_t f = 0; f < features.size(); f++){
           ft[f] = features[f]->getValue(idx);
        }

        booster->Predict(ft.data(), probs.data(), &earlyStop);
        const auto maxClass = std::max_element(probs.begin(), probs.end());
        int bestClass = std::distance(probs.begin(), maxClass);

        auto label = labels[bestClass];
        if (useColors){
            auto color = label.getColor();
            pointSet.colors[i][0] = color.r;
            pointSet.colors[i][1] = color.g;
            pointSet.colors[i][2] = color.b;
        }else{
            // TODO
        }

        if (evaluate && pointSet.labels[i] == bestClass){
            correct++;
        }
    }

    if (evaluate){
        float modelErr = (1.f - static_cast<float>(correct) / static_cast<float>(pointSet.count()));
        std::cout << "Model error: " << std::setprecision(4) << (modelErr * 100.f) << "%" << std::endl;
    }
}

}

