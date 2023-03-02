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
    std::vector< std::vector<double> > featuresData(features.size());
    std::vector< std::vector<int> > featuresIdx(features.size());

    std::vector<std::size_t> count (labels.size(), 0);
    // std::vector<double> fs(features.size());
    // std::vector<int> fx(features.size());

    size_t c = 0;
    for (size_t i = 0; i < pointSet.count(); i++){
        int g = pointSet.labels[i];
        if (g != LABEL_UNASSIGNED) {
            gt.push_back(g);
            for (std::size_t f = 0; f < features.size(); f++){
                featuresData[f].push_back(features[f]->getValue(i));
                featuresIdx[f].push_back(c++);
            }
            count[std::size_t(g)]++;
        }
    }

    std::cout << "Using " << gt.size() << " inliers:" << std::endl;
    for (std::size_t i = 0; i < labels.size(); i++)
        std::cout << " * " << labels[i].getName() << ": " << count[i] << std::endl;


    const size_t numRows = gt.size();
    const size_t numFeats = features.size();
    const int numClasses = static_cast<int>(labels.size());

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
    
    exit(1); // WHY IS IT CRASHING?
    
    //#pragma omp parallel
    {
        std::vector<double> row(features.size());

        //#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < numRows; ++i){
            for (size_t f = 0; f < features.size(); f++)
                row[f] = featuresData[f][i];
            
            dset->PushOneRow(omp_get_thread_num(), i, row);
        }
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
    boostConfig.num_iterations = 20;
    // boostConfig.bagging_freq = 1;
    boostConfig.bagging_fraction = 0.5;
    boostConfig.num_class = numClasses;

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

    // Validation stuff

    std::vector<int> predictedClass(numRows);
    std::vector<double> predVec(numClasses);

    LightGBM::PredictionEarlyStopConfig early_stop_config;
    auto earlyStop = LightGBM::CreatePredictionEarlyStopInstance("none", early_stop_config);

    booster->InitPredict(booster->GetCurrentIteration() - 1, 0, false);

    for (int i=0; i < numRows; i++){
        booster->Predict(featuresData[i].data(), predVec.data(), &earlyStop);
        const auto predMax = std::max_element(predVec.begin(), predVec.end());
        predictedClass[i] = std::distance(predVec.begin(), predMax);
    }

    // compute error
    double err = 0;
    for (int i = 0; i < numRows; i++) {
        if (predictedClass[i] != gt[i]){
            err++;
        }
    }
    err /= numRows;

    std::cout << "Training error: " << (err * 100) << "%" << std::endl;

    exit(1);
/*
    liblearning::DataView2D<int> label_vector (&(gt[0]), gt.size(), 1);
    liblearning::DataView2D<float> feature_vector(&(ft[0]), gt.size(), ft.size() / gt.size());

    std::cout << "Training..." << std::endl;
    rtrees.train(feature_vector, label_vector, liblearning::DataView2D<int>(), generator, 0, false);
    
    std::ofstream ofs(modelFilename.c_str(), std::ios_base::out | std::ios_base::binary | std::ios_base::trunc);
    boost::iostreams::filtering_ostream outs;
    outs.push(boost::iostreams::gzip_compressor());
    outs.push(ofs);
    boost::archive::text_oarchive oas(outs);
    oas << BOOST_SERIALIZATION_NVP(rtrees);

    // TODO: copy what CGAL is doing

    std::cout << "Saved " << modelFilename << std::endl;
    
    exit(1);*/
}

void classify(PointSet &pointSet, 
    const std::string &modelFilename,
    const std::vector<Feature *> &features, 
    const std::vector<Label> &labels,
    bool useColors,
    bool evaluate){
    
    
    exit(1);
}

}

