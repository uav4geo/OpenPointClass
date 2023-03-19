#include <iostream>
#include <random>
#include <algorithm>

#include "classifier.hpp"
#include "gbm.hpp"

namespace gbm{


Boosting* train(const std::vector<std::string> filenames,
    double *startResolution,
    int numScales,
    int numTrees,
    int treeDepth,
    double radius,
    int maxSamples){

    std::vector<float> gt;
    std::vector< std::vector<double> > featureRows;
    std::vector< std::vector<double> > featuresData;
    std::vector< std::vector<int> > featuresIdx;
    size_t numFeats;
    int numClass;

    getTrainingData(filenames, startResolution, numScales, radius, maxSamples,
        [&featureRows, &featuresData, &featuresIdx, &gt](std::vector<Feature *> &features, size_t idx, int g){
            size_t row = featureRows.size();
            featureRows.emplace_back();
            featureRows[row].resize(features.size(), 0);
            for (std::size_t f = 0; f < features.size(); f++){
                featureRows[row][f] = features[f]->getValue(idx);
                featuresData[f].push_back(featureRows[row][f]);
                featuresIdx[f].push_back(row);
            }
            gt.push_back(g);
        },
        [&numFeats, &numClass, &featuresData, &featuresIdx](size_t numFeatures, int numClasses){
            numFeats = numFeatures;
            numClass = numClasses;
            featuresData.resize(numFeats);
            featuresIdx.resize(numFeats);
        });

    const size_t numRows = gt.size();
    std::cout << "Using " << numRows << " inliers" << std::endl;

    LightGBM::Config ioconfig;
    ioconfig.num_class = numClass;
    ioconfig.max_bin = 255;

    std::unique_ptr<LightGBM::Dataset> dset;
    LightGBM::DatasetLoader loader(ioconfig, nullptr, numClass, nullptr);
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

/*
    for(int j = 0; j < numFeats; j++){
        const auto nbins = dset->FeatureBinMapper(j)->num_bin();
        // std::cout << features[j]->getName() << std::endl;
        std::cout << "Feat " << j << std::endl;
        std::cout << "   " << dset->FeatureBinMapper(j)->BinToValue(0) << " ";
        std::cout << "   " << dset->FeatureBinMapper(j)->BinToValue(nbins-2) << " ";
        std::cout << std::endl;
    }
*/

    if (!dset->SetFloatField("label", gt.data(), numRows)) {
        throw std::runtime_error("Error setting label");
    }

    LightGBM::Config boostConfig;
    boostConfig.num_iterations = numTrees;
    // boostConfig.bagging_freq = 1;
    boostConfig.bagging_fraction = 0.5;
    boostConfig.num_class = numClass;
    boostConfig.max_depth = treeDepth;

    // tree params
    boostConfig.num_leaves = 16;
    boostConfig.learning_rate = 0.2;

    std::stringstream ss;
    ss << *startResolution << " " <<  radius << " " << numScales;
    boostConfig.data = ss.str();

    LightGBM::Config objConfig;
    objConfig.num_class = numClass;

    auto *objFunc = LightGBM::ObjectiveFunction::CreateObjectiveFunction("multiclass", objConfig);
    objFunc->Init(dset->metadata(), dset->num_data());

    LightGBM::Config metricConfig;
    metricConfig.num_class = numClass;

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
        auto scores = booster->GetEvalAt(0);
        for(auto &v: scores) std::cout << "Iteration " << (i+1) << " score: " << v << std::endl;

        if (booster->TrainOneIter(nullptr, nullptr)){
            std::cout << "Breaking.." << std::endl;
            break;
        }
    }

    return booster;
}

Boosting *loadBooster(const std::string &modelFilename){
    std::cout << "Loading " << modelFilename << std::endl;

    auto *booster = LightGBM::Boosting::CreateBoosting("gbdt", nullptr);
    if (!LightGBM::Boosting::LoadFileToBoosting(booster, modelFilename.c_str())){
        throw std::runtime_error("Cannot open " + modelFilename);
    }
    booster->InitPredict(0, 0, false);

    return booster;
}

void saveBooster(Boosting *booster, const std::string &modelFilename){
    booster->SaveModelToFile(0, 0, 0, modelFilename.c_str());
    std::cout << "Saved " << modelFilename << std::endl;
}

BoosterParams extractBoosterParams(Boosting *booster){
    json j = json::parse(booster->GetLoadedParam());
    if (!j.contains("data") || j["data"].get<std::string>().empty()) throw std::runtime_error("Invalid booster model (data params missing?)");

    std::stringstream ss(j["data"].get<std::string>());
    BoosterParams p;
    ss >> p.resolution;
    ss >> p.radius;
    ss >> p.numScales;
    return p;
    
}

void classify(PointSet &pointSet, 
    Boosting *booster,
    const std::vector<Feature *> &features, 
    const std::vector<Label> &labels,
    Regularization regularization,
    double regRadius,
    bool useColors,
    bool unclassifiedOnly,
    bool evaluate,
    const std::vector<int> &skip){

    LightGBM::PredictionEarlyStopConfig early_stop_config;
    auto earlyStop = LightGBM::CreatePredictionEarlyStopInstance("none", early_stop_config);

    classifyData<double>(pointSet, 
      [&booster, &earlyStop](double *ft, double *probs){
          booster->Predict(ft, probs, &earlyStop);
      },
      features, labels, regularization, regRadius, useColors, unclassifiedOnly, evaluate, skip);
}

}

