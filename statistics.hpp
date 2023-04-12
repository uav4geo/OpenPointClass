
#ifndef STATISTICS_H
#define STATISTICS_H

#include <vector>
#include <map>

// Accuracy = (True Positives + True Negatives) / (True Positives + True Negatives + False Positives + False Negatives)
// Precision = True Positives / (True Positives + False Positives)
// recall = Recall = True Positives / (True Positives + False Negatives)
// F1 score = 2 * (Precision * recall) / (Precision + recall)
// Specificity = True Negatives / (True Negatives + False Positives))
// Intersection over Union = True Positives / (True Positives + False Positives + False Negatives)

// true positives, false positives, true negatives, false negatives

class Statistics{
    struct Counts{
        size_t tp = 0;
        size_t fp = 0;
        size_t fn = 0;
    };

    struct LabelStat{
        std::string name;
        double accuracy;
        double recall;
        double precision;
        double f1;
        Counts counts;

        LabelStat(const std::string &name, const double accuracy, const double recall, const double precision, const double f1, const Counts &counts) : 
            name(name), accuracy(accuracy), recall(recall), precision(precision), f1(f1), counts(counts) {}
    };

    std::map<int, Counts> stats;
    const std::vector<Label> &labels;
    int totalSamples = 0;

    double accuracy;
    double avgAccuracy;
    double avgRecall;
    double avgPrecision;
    double avgF1;

    std::vector<LabelStat> labelStats;
public:
    Statistics(const std::vector<Label> &labels) : labels(labels){
        for (auto &label : labels){
            stats[label.getTrainingCode()] = Counts();
        }
    }

    inline void record(const int predicted, const int truth){
        if (predicted == truth){
            #pragma omp atomic
            stats[predicted].tp++;
        }else{
            #pragma omp atomic
            stats[predicted].fp++;

            #pragma omp atomic
            stats[truth].fn++;
        }

        #pragma omp atomic
        totalSamples++;
    }

    void finalize(){
        const auto cnt = labels.size();

        auto accuracyCount = 0;
        auto recallCount = 0;
        auto precisionCount = 0;
        auto f1Count = 0;

        int sumTp = 0;

        double sumAccuracy = 0.0;
        double sumF1 = 0.0;
        double sumRecall = 0.0;
        double sumPrecision = 0.0;

        for (size_t i = 0; i < cnt; ++i){
            auto label = labels[i];
            const Counts &cnts = stats[label.getTrainingCode()];
            sumTp += cnts.tp;

            const auto tp = static_cast<double>(cnts.tp);

            const double accuracy = tp / (cnts.tp + cnts.fn + cnts.fp);

            if (!std::isnan(accuracy)) {
                sumAccuracy += accuracy;
                accuracyCount++;
            }

            const double precision = tp / (cnts.tp + cnts.fp);

            if (!std::isnan(precision)) {
                sumPrecision += precision;
                precisionCount++;
            }

            const double recall = tp / (cnts.tp + cnts.fn);

            if (!std::isnan(recall)) {
                sumRecall += recall;
                recallCount++;
            }

            const double f1 = 2 * (precision * recall) / (precision + recall);

            if (!std::isnan(f1)) {
                sumF1 += f1;
                f1Count++;
            }

            labelStats.emplace_back(label.getName(), accuracy, recall, precision, f1, cnts);
        }

        accuracy = static_cast<double>(sumTp) / totalSamples;

        avgAccuracy = sumAccuracy / accuracyCount;
        avgF1 = sumF1 / f1Count;
        avgPrecision = sumPrecision / precisionCount;
        avgRecall = sumRecall / recallCount;
    }

    void print() const{
        std::cout << "Statistics:" << std::endl;
        std::cout << "  Accuracy: " << std::fixed << std::setprecision(2) << accuracy * 100 << "%" << std::endl << std::endl;

        std::cout << "  " << std::setw(24) << "Label " << " | " << std::setw(10) << "Accuracy" << " | " << std::setw(11) << "Recall" << " | " << std::setw(10) << "Precision" << " | " << std::setw(10) << "F1" << " | "  << std::endl;
        std::cout << "  " << std::setw(24) << std::string(24, '-') << " | ";
        std::cout << std::setw(10) << std::string(10, '-') << " | ";
        std::cout << std::setw(10) << std::string(11, '-') << " | ";
        std::cout << std::setw(10) << std::string(10, '-') << " | ";
        std::cout << std::setw(10) << std::string(10, '-') << " | "  << std::endl;

        for (const auto &label : labelStats){

            if (std::isnan(label.accuracy) && std::isnan(label.f1)) continue;

            std::cout << "  " << std::setw(24) << label.name << " | ";

            if (!std::isnan(label.accuracy))
                std::cout << std::setw(9) << std::fixed << std::setprecision(2) << label.accuracy * 100 << "% | ";
            else
                std::cout << std::setw(9) << "N/A" << " | ";

            if (!std::isnan(label.recall))
                std::cout << std::setw(10) << std::fixed << std::setprecision(2) << label.recall * 100 << "% | ";
            else
                std::cout << std::setw(10) << "N/A" << " | ";

            if (!std::isnan(label.precision))
                std::cout << std::setw(9) << std::fixed << std::setprecision(2) << label.precision * 100 << "% | ";
            else
                std::cout << std::setw(9) << "N/A" << " | ";
            
            if (!std::isnan(label.f1))
                std::cout << std::setw(10) << std::fixed << std::setprecision(2) << label.f1 << " | ";
            else
                std::cout << std::setw(10) << "N/A" << " | ";

            std::cout << std::endl;
        }

        std::cout << "  " << std::setw(24) << "(Average)" << " | ";
        std::cout << std::setw(9) << std::fixed << std::setprecision(2) << avgAccuracy * 100 << "% | ";
        std::cout << std::setw(10) << std::fixed << std::setprecision(2) << avgRecall * 100 << "% | ";
        std::cout << std::setw(9) << std::fixed << std::setprecision(2) << avgPrecision * 100 << "% | ";
        std::cout << std::setw(10) << std::fixed << std::setprecision(2) << avgF1 << " | " << std::endl;

        std::cout << std::endl;
    }


    void writeToFile(const std::string &jsonFile){
        std::ofstream o(jsonFile);
        if (!o.is_open()){
            std::cerr << "Unable to create stats file" << std::endl;
            return;
        }

        json j = json{
            {"accuracy", accuracy}
        };

        for (const auto &label : labelStats){
            if (std::isnan(label.accuracy) && std::isnan(label.f1)) continue;
            const auto name = label.name;

            if (!std::isnan(label.accuracy))
                j["labels"][name]["accuracy"] = label.accuracy;
            else
                j["labels"][name]["accuracy"] = nullptr;

            if (!std::isnan(label.precision))
                j["labels"][name]["precision"] = label.precision;
            else
                j["labels"][name]["precision"] = nullptr;

            if (!std::isnan(label.recall))
                j["labels"][name]["recall"] = label.recall;
            else
                j["labels"][name]["recall"] = nullptr;

            if (!std::isnan(label.f1))
                j["labels"][name]["f1"] = label.f1;
            else
                j["labels"][name]["f1"] = nullptr;
        }

        o << j.dump(4);
        o.close();
        std::cout << "Statistics saved to " << jsonFile << std::endl;
    }
};


#endif
