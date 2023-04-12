#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include <vector>
#include <random>
#include <cmath>

#include "features.hpp"
#include "labels.hpp"
#include "constants.hpp"
#include "point_io.hpp"
#include "statistics.hpp"

enum Regularization { None, LocalSmooth };
Regularization parseRegularization(const std::string &regularization);

enum ClassifierType { RandomForest, GradientBoostedTrees };
ClassifierType fingerprint(const std::string &modelFile);


template <typename F, typename I>
void getTrainingData(const std::vector<std::string> &filenames,
    double *startResolution,
    const int numScales,
    const double radius,
    const int maxSamples,
    const std::vector<int> &asprsClasses,
    F storeFeatures,
    I init) {
    const auto labels = getTrainingLabels();

    const bool trainSubset = asprsClasses.size() > 0;
    std::array<bool, 255> trainClass;

    if (trainSubset) {
        trainClass.fill(false);

        auto asprsToTrain = getAsprs2TrainCodes();
        for (auto &c : asprsClasses) {
            trainClass[asprsToTrain[c]] = true;
        }
    }

    for (size_t i = 0; i < filenames.size(); i++) {
        std::cout << "Processing " << filenames[i] << std::endl;
        auto pointSet = readPointSet(filenames[i]);
        if (!pointSet->hasLabels()) {
            std::cout << filenames[i] << " has no labels, skipping..." << std::endl;
            continue;
        }

        if (*startResolution == -1.0) {
            *startResolution = pointSet->spacing(); // meters
            std::cout << "Starting resolution: " << *startResolution << std::endl;
        }

        auto scales = computeScales(numScales, pointSet, *startResolution, radius);
        auto features = getFeatures(scales);
        std::cout << "Features: " << features.size() << std::endl;

        if (i == 0) init(features.size(), labels.size());

        std::vector<std::size_t> count(labels.size(), 0);
        std::vector<bool> sampled(pointSet->count(), false);
        std::vector<std::pair<size_t, int> > idxes;

        for (size_t i = 0; i < pointSet->count(); i++) {
            int g = pointSet->labels[i];
            if (g != LABEL_UNASSIGNED) {
                if (trainSubset && !trainClass[g]) continue;

                size_t idx = pointSet->pointMap[i];
                if (!sampled[idx]) {
                    idxes.push_back(std::make_pair(idx, g));
                    count[static_cast<std::size_t>(g)]++;
                    sampled[idx] = true;
                }
            }
        }

        size_t samplesPerLabel = std::numeric_limits<size_t>::max();
        for (std::size_t i = 0; i < labels.size(); i++) {
            if (count[i] > 0) samplesPerLabel = std::min(count[i], samplesPerLabel);
        }
        samplesPerLabel = std::min<size_t>(samplesPerLabel, maxSamples);
        std::vector<std::size_t> added(labels.size(), 0);

        std::cout << "Samples per label: " << samplesPerLabel << std::endl;

        std::random_device rd;
        std::mt19937 ranGen(rd());
        std::shuffle(idxes.begin(), idxes.end(), ranGen);

        for (const auto &p : idxes) {
            size_t idx = p.first;
            int g = p.second;
            if (added[static_cast<std::size_t>(g)] < samplesPerLabel) {
                storeFeatures(features, idx, g);
                added[static_cast<std::size_t>(g)]++;
            }
        }

        for (std::size_t i = 0; i < labels.size(); i++)
            std::cout << " * " << labels[i].getName() << ": " << added[i] << " / " << count[i] << std::endl;

        // Free up memory for next
        for (size_t i = 0; i < scales.size(); i++) delete scales[i];
        for (size_t i = 0; i < features.size(); i++) delete features[i];
        RELEASE_POINTSET(pointSet);
    }
}

template <typename T, typename F>
void classifyData(PointSet &pointSet,
    F evaluateFunc,
    const std::vector<Feature *> &features,
    const std::vector<Label> &labels,
    const Regularization regularization,
    const double regRadius,
    const bool useColors,
    const bool unclassifiedOnly,
    const bool evaluate,
    const std::vector<int> &skip,
    const std::string &statsFile) {

    std::cout << "Classifying..." << std::endl;
    pointSet.base->labels.resize(pointSet.base->count());

    if (regularization == Regularization::None) {
        #pragma omp parallel
        {
            std::vector<T> probs(labels.size(), 0.);
            std::vector<T> ft(features.size());

            #pragma omp for
            for (long long int i = 0; i < pointSet.base->count(); i++) {
                for (std::size_t f = 0; f < features.size(); f++) {
                    ft[f] = features[f]->getValue(i);
                }

                evaluateFunc(ft.data(), probs.data());

                // Find highest probability
                int bestClass = 0;
                T bestClassVal = 0.;

                for (std::size_t j = 0; j < probs.size(); j++) {
                    if (probs[j] > bestClassVal) {
                        bestClass = j;
                        bestClassVal = probs[j];
                    }
                }

                pointSet.base->labels[i] = bestClass;
            }
        } // end pragma omp

    }
    else if (regularization == Regularization::LocalSmooth) {
        std::vector<std::vector<T> > values(labels.size(), std::vector<T>(pointSet.base->count(), -1.));

        #pragma omp parallel
        {

            std::vector<T> probs(labels.size(), 0.);
            std::vector<T> ft(features.size());

            #pragma omp for
            for (long long int i = 0; i < pointSet.base->count(); i++) {
                for (std::size_t f = 0; f < features.size(); f++) {
                    ft[f] = features[f]->getValue(i);
                }

                evaluateFunc(ft.data(), probs.data());

                for (std::size_t j = 0; j < labels.size(); j++) {
                    values[j][i] = probs[j];
                }
            }

        }

        std::cout << "Local smoothing..." << std::endl;

        #pragma omp parallel
        {

            std::vector<nanoflann::ResultItem<size_t, float>> radiusMatches;
            std::vector<T> mean(values.size(), 0.);
            auto index = pointSet.base->getIndex<KdTree>();

            #pragma omp for schedule(dynamic, 1)
            for (long long int i = 0; i < pointSet.base->count(); i++) {
                size_t numMatches = index->radiusSearch(&pointSet.base->points[i][0], regRadius, radiusMatches);
                std::fill(mean.begin(), mean.end(), 0.);

                for (size_t n = 0; n < numMatches; n++) {
                    for (std::size_t j = 0; j < values.size(); ++j) {
                        mean[j] += values[j][radiusMatches[n].first];
                    }
                }

                int bestClass = 0;
                T bestClassVal = 0.f;
                for (std::size_t j = 0; j < mean.size(); j++) {
                    mean[j] /= numMatches;
                    if (mean[j] > bestClassVal) {
                        bestClassVal = mean[j];
                        bestClass = j;
                    }
                }

                pointSet.base->labels[i] = bestClass;
            }

        }
    }
    else {
        throw std::runtime_error("Invalid regularization");
    }

    //std::size_t correct = 0;
    if (!useColors && !pointSet.hasLabels()) pointSet.labels.resize(pointSet.count());
    std::vector<bool> skipMap(255, false);
    for (size_t i = 0; i < skip.size(); i++) {
        int skipClass = skip[i];
        if (skipClass >= 0 && skipClass <= 255) skipMap[skipClass] = true;
    }

    auto train2asprsCodes = getTrain2AsprsCodes();

    std::vector<int> trainCodes(labels.size());
    for (size_t i = 0; i < labels.size(); i++)
    {
        trainCodes[i] = labels[i].getTrainingCode();
    }

    Statistics stats(trainCodes);

    #pragma omp parallel for
    for (long long int i = 0; i < pointSet.count(); i++) {
        size_t idx = pointSet.pointMap[i];

        int bestClass = pointSet.base->labels[idx];
        auto label = labels[bestClass];

        if (evaluate) {
            stats.updateStatistics(bestClass, pointSet.labels[i]);
        }

        bool update = true;
        bool hasLabels = pointSet.hasLabels();

        // if unclassifiedOnly, do not update points with an existing classification
        if (unclassifiedOnly && hasLabels
            && pointSet.labels[i] != LABEL_UNCLASSIFIED) update = false;

        const int asprsCode = label.getAsprsCode();
        if (skipMap[asprsCode]) update = false;

        if (update) {
            if (useColors) {
                const auto color = label.getColor();
                pointSet.colors[i][0] = color.r;
                pointSet.colors[i][1] = color.g;
                pointSet.colors[i][2] = color.b;
            }
            else {
                pointSet.labels[i] = asprsCode;
            }
        }
        else if (hasLabels) {
            // We revert training codes back to ASPRS
            pointSet.labels[i] = train2asprsCodes[pointSet.labels[i]];
        }
    }

    if (evaluate) {

        auto [globalAccuracy, labelsAccuracy, avgAccuracy, f1Scores, avgF1] = stats.getStatistics();

        std::cout << "Statistics:" << std::endl;
        std::cout << "  Global accuracy: " << std::fixed << std::setprecision(3) << globalAccuracy * 100 << "%" << std::endl;
        std::cout << "  Average accuracy: " << std::fixed << std::setprecision(3) << avgAccuracy * 100 << "%" << std::endl;
        std::cout << "  Average F1: " << std::fixed << std::setprecision(3) << avgF1 << std::endl;
        std::cout << "  Labels:" << std::endl << std::endl;

        std::cout << "  " << std::setw(25) << "Label " << " | " << std::setw(10) << "Accuracy" << " | " << std::setw(10) << "F1" << std::endl;
        std::cout << "  " << std::setw(25) << std::string(25, '-') << " | " << std::setw(10) << std::string(10, '-') << " | " << std::setw(10) << std::string(10, '-') << std::endl;

        for (auto n = 0; n < labels.size(); n++)
        {

            if (std::isnan(labelsAccuracy[n]) && std::isnan(f1Scores[n])) continue;

            std::cout << "  " << std::setw(25) << labels[n].getName() << " | ";

            if (!std::isnan(labelsAccuracy[n]))
                std::cout << std::setw(9) << std::fixed << std::setprecision(3) << labelsAccuracy[n] * 100 << "% | ";
            else
                std::cout << std::setw(10) << "N/A" << " | ";


            if (!std::isnan(f1Scores[n]))
                std::cout << std::setw(10) << std::fixed << std::setprecision(3) << f1Scores[n];
            else
                std::cout << std::setw(10) << "N/A";

            std::cout << std::endl;

        }

        std::cout << std::endl;

        if (!statsFile.empty())
        {
            std::ofstream o(statsFile);

            auto j = json {
                {"globalAccuracy", globalAccuracy},
                {"avgAccuracy", avgAccuracy},
                {"avgF1", avgF1}
            };

            for (auto n = 0; n < labels.size(); n++)
            {

                if (std::isnan(labelsAccuracy[n]) && std::isnan(f1Scores[n])) continue;
                const auto name = labels[n].getName();
                if (!std::isnan(labelsAccuracy[n]))
                    j["labels"][name]["accuracy"] = labelsAccuracy[n];
                else
                    j["labels"][name]["accuracy"] = nullptr;

                if (!std::isnan(f1Scores[n]))
                    j["labels"][name]["f1"] = f1Scores[n];
                else
                    j["labels"][name]["f1"] = nullptr;
            }

            if (o.is_open()) {
                o << j.dump(4);
                o.close();
                std::cout << "Statistics saved to " << statsFile << std::endl;
            }
            else {
                std::cerr << "Unable to create stats file" << std::endl;
            }
        }

    }
}

#endif

