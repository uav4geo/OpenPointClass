#include <LightGBM/config.h>
#include <LightGBM/dataset_loader.h>
#include <LightGBM/boosting.h>
#include <LightGBM/objective_function.h>
#include <LightGBM/metric.h>
#include <LightGBM/utils/common.h>

#include <iostream>
#include <random>
#include <algorithm>

#include "gbm.hpp"

namespace gbm{

void test(){
/*
    std::random_device rd;
    std::mt19937 gen(rd());

    // one random generator for every class
    std::vector<std::normal_distribution<>> dists = {
        std::normal_distribution<>(0, 1),
        std::normal_distribution<>(10, 1)};

    const int numSamples = 5000;
    const int numFeats = 2;
    const int numClasses = static_cast<int>(dists.size());

    std::cout << "Num classes: " << numClasses << std::endl;

    // labels
    std::vector<float>   labels(numSamples);
    for (int i=0; i < numSamples; i++)
        labels[i] = i % numClasses;

    std::vector< std::vector<double> > features(numSamples);
    for (int i=0; i < numSamples; i++)
    {
        features[i].resize(numFeats);
        for (int j=0; j < numFeats; j++)
        {
            const auto lbl = static_cast<int>(labels[i]);
            features[i][j] = dists[lbl](gen);
        }
    }

    // prepare sample data
    std::vector< std::vector<double> > sampleData(numFeats);
    std::vector< std::vector<int> > sampleIndices(numFeats);

    for (int i=0; i < numSamples; i++)
    {
        for (int j=0; j < numFeats; j++){
            sampleData[j].push_back(features[i][j]);
            sampleIndices[j].push_back(i);
        }
    }

    LightGBM::Config config;
    config.num_class = numClasses;
    config.max_bin = 255;
    config.verbosity = 10;

    std::unique_ptr<LightGBM::Dataset> dset;

    LightGBM::DatasetLoader loader(config, nullptr, numClasses, nullptr);
    dset.reset( loader.ConstructFromSampleData(sampleData.data(), sampleIndices.data(),  numSamples, numSamples, numSamples) );
    
    for (int i = 0; i < numSamples; ++i){
        const int thread_id = 0;
        dset->PushOneRow(thread_id, i, features[i]);
    }

    dset->FinishLoad();

    // check bins
    for(int j=0; j < numFeats; j++)
    {
        const auto nbins = dset->FeatureAt(j)->bin_mapper()->num_bin();
        std::cout << "Feat " << numFeats << std::endl;
        std::cout << "   " << dset->FeatureAt(j)->bin_mapper()->BinToValue(0) << " ";
        std::cout << "   " << dset->FeatureAt(j)->bin_mapper()->BinToValue(nbins-2) << " ";

        std::cout << std::endl;
    }

    if (!dset->SetFloatField("label", labels.data(), numSamples)) {
        std::cout << "Error setting label" << std::endl;
        return -1;
    }


    LightGBM::BoostingConfig boostConfig;
    boostConfig.num_iterations = 100;
    boostConfig.bagging_freq = 1;
    boostConfig.bagging_fraction = 0.5;
    boostConfig.num_class = numClasses;

    // tree params
    boostConfig.tree_config.min_data_in_leaf = 10;
    boostConfig.tree_config.num_leaves = 16;
    //boostConfig.tree_config.min_sum_hessian_in_leaf = 0;

    LightGBM::ObjectiveConfig objConfig;
    objConfig.num_class = numClasses;
    // objConfig.label_gain.clear();
    // objConfig.label_gain.resize(numClasses, 1.0);

    auto *objFunc = LightGBM::ObjectiveFunction::CreateObjectiveFunction("multiclass", objConfig);
    objFunc->Init(dset->metadata(), dset->num_data());

    LightGBM::MetricConfig metricConfig;
    metricConfig.num_class = numClasses;

    std::vector< std::unique_ptr<LightGBM::Metric> > trainMetrics;
    auto metric = std::unique_ptr<LightGBM::Metric>(
        LightGBM::Metric::CreateMetric("multi_logloss", metricConfig));
    metric->Init(dset->metadata(), dset->num_data());
    trainMetrics.push_back(std::move(metric));

    auto *booster = LightGBM::Boosting::CreateBoosting(LightGBM::BoostingType::kGBDT, nullptr);

    booster->Init(&boostConfig, nullptr, objFunc,
        LightGBM::Common::ConstPtrInVectorWrapper<LightGBM::Metric>(trainMetrics));

    booster->ResetTrainingData(&boostConfig, dset.get(), objFunc,
        LightGBM::Common::ConstPtrInVectorWrapper<LightGBM::Metric>(trainMetrics));

    // booster->AddValidDataset(dset.get(), LightGBM::Common::ConstPtrInVectorWrapper<LightGBM::Metric>(trainMetrics));

    for (int i=0; i < boostConfig.num_iterations; i++)
    {
        std::cout << "Iteration " << (i+1) << std::endl;

        auto scores = booster->GetEvalAt(0);
        for(auto &v: scores)
            std::cout << "Score: " << v << std::endl;

        if (booster->TrainOneIter(nullptr, nullptr))
        {
            std::cout << "Breaking.." << std::endl;
            break;
        }
    }

    booster->SetNumIterationForPred(0); // predict with all trees

    std::vector<int> predictedClass(numSamples);
    for (int i=0; i < numSamples; i++)
    {
        auto predVec = booster->PredictRaw(features[i].data());
        const auto predMax = std::max_element(predVec.begin(), predVec.end());

        predictedClass[i] = std::distance(predVec.begin(), predMax);
    }

    // compute error
    double err = 0;
    for (int i=0; i < numSamples; i++)
    {
        if (predictedClass[i] != labels[i])
        {
            err++;
        }
    }
    err /= labels.size();

    std::cout << "Training error: " << err << std::endl;
*/
    exit(1);
}

void train(const PointSet &pointSet, 
    const std::vector<Feature *> &features, 
    const std::vector<Label> &labels,
    const std::string &modelFilename){


    
    exit(1);
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

