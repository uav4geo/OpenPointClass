
#ifndef STATISTICS_H
#define STATISTICS_H

#include <vector>
#include <map>

// Accuracy = (True Positives + True Negatives) / (True Positives + True Negatives + False Positives + False Negatives)
// Precision = True Positives / (True Positives + False Positives)
// Sensitivity = Recall = True Positives / (True Positives + False Negatives)
// F1 score = 2 * (Precision * Sensitivity) / (Precision + Sensitivity)
// Specificity = True Negatives / (True Negatives + False Positives))
// Intersection over Union = True Positives / (True Positives + False Positives + False Negatives)

// true positives, false positives, true negatives, false negatives


struct Counts
{

    Counts()
    {
        tp = 0;
        fp = 0;
        fn = 0;
    }

    int tp;
    int fp;
    int fn;

};


struct LabelStat
{
    std::string name;
    double accuracy;
    double sensitivity;
    double precision;
    double f1;

    Counts counts;

    LabelStat(const std::string &name, const double accuracy, const double sensitivity, const double precision, const double f1, const Counts &counts)
    {
        this->accuracy = accuracy;
        this->sensitivity = sensitivity;
        this->precision = precision;
        this->f1 = f1;
        this->counts = counts;
        this->name = name;
    }

};

struct Stat
{
    double globalAccuracy;

    double avgAccuracy;
    double avgSensitivity;
    double avgPrecision;
    double avgF1;

    std::vector<LabelStat> labels;

    void print() const
    {
        std::cout << "Statistics:" << std::endl;
        std::cout << "  Global accuracy: " << std::fixed << std::setprecision(3) << globalAccuracy * 100 << "%" << std::endl;

        std::cout << "  Average accuracy: " << std::fixed << std::setprecision(3) << avgAccuracy * 100 << "%" << std::endl;
        std::cout << "  Average sensitivity: " << std::fixed << std::setprecision(3) << avgSensitivity * 100 << "%" << std::endl;
        std::cout << "  Average precision: " << std::fixed << std::setprecision(3) << avgPrecision * 100 << "%" << std::endl;
        std::cout << "  Average F1: " << std::fixed << std::setprecision(3) << avgF1 << std::endl;
        std::cout << std::endl << "  Labels:" << std::endl << std::endl;

        std::cout << "  " << std::setw(25) << "Label " << " | " << std::setw(10) << "Accuracy" << " | " << std::setw(11) << "Sensitivity" << " | " << std::setw(10) << "Precision" << " | " << std::setw(10) << "F1" << " | "  << std::endl;
        std::cout << "  " << std::setw(25) << std::string(25, '-') << " | ";
        std::cout << std::setw(10) << std::string(10, '-') << " | ";
        std::cout << std::setw(11) << std::string(10, '-') << " | ";
        std::cout << std::setw(10) << std::string(10, '-') << " | ";
        std::cout << std::setw(10) << std::string(10, '-') << " | "  << std::endl;

        for (const auto &label : labels)
        {

            if (std::isnan(label.accuracy) && std::isnan(label.f1)) continue;

            std::cout << "  " << std::setw(25) << label.name << " | ";

            if (!std::isnan(label.accuracy))
                std::cout << std::setw(9) << std::fixed << std::setprecision(3) << label.accuracy * 100 << "% | ";
            else
                std::cout << std::setw(10) << "N/A" << " | ";

            if (!std::isnan(label.sensitivity))
                std::cout << std::setw(11) << std::fixed << std::setprecision(3) << label.sensitivity << " | ";
            else
                std::cout << std::setw(11) << "N/A" << " | ";

            if (!std::isnan(label.precision))
                std::cout << std::setw(10) << std::fixed << std::setprecision(3) << label.precision << " | ";
            else
                std::cout << std::setw(10) << "N/A" << " | ";
            
            if (!std::isnan(label.f1))
                std::cout << std::setw(10) << std::fixed << std::setprecision(3) << label.f1 << " | ";
            else
                std::cout << std::setw(10) << "N/A" << " | ";

            std::cout << std::endl;

        }

        std::cout << std::endl;
    }

    void to_json(json &j) {
        j = json{
                    {"globalAccuracy", globalAccuracy},
                    {"avgAccuracy", avgAccuracy},
                    {"avgSensitivity", avgSensitivity},
                    {"avgPrecision", avgPrecision},
                    {"avgF1", avgF1}
        };

        for (const auto &label : labels)
        {

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

            if (!std::isnan(label.sensitivity))
                j["labels"][name]["sensitivity"] = label.sensitivity;
            else
                j["labels"][name]["sensitivity"] = nullptr;

            if (!std::isnan(label.f1))
                j["labels"][name]["f1"] = label.f1;
            else
                j["labels"][name]["f1"] = nullptr;
        }
    }
};



class Statistics
{

    std::map<int, Counts> stats;
    const std::vector<Label> &labels;
    int totalSamples = 0;

public:

    Statistics(const std::vector<Label> &labels) : labels(labels)
    {
        for (auto &label : labels)
        {
            stats[label.getTrainingCode()] = Counts();
        }
    }

    void updateStatistics(const int predicted, const int truth)
    {
        if (predicted == truth)
        {
            #pragma omp atomic
            stats[predicted].tp++;
        }
        else
        {
            #pragma omp atomic
            stats[predicted].fp++;

            #pragma omp atomic
            stats[truth].fn++;
        }

        #pragma omp atomic
        totalSamples++;
    }


    Stat getStatistics()
    {
        Stat result;

        const auto cnt = labels.size();

        auto accuracyCount = 0;
        auto sensitivityCount = 0;
        auto precisionCount = 0;
        auto f1Count = 0;

        int sumTp = 0;

        double sumAccuracy = 0.0;
        double sumF1 = 0.0;
        double sumSensitivity = 0.0;
        double sumPrecision = 0.0;

        for (size_t i = 0; i < cnt; ++i)
        {
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

            const double sensitivity = tp / (cnts.tp + cnts.fn);

            if (!std::isnan(sensitivity)) {
                sumSensitivity += sensitivity;
                sensitivityCount++;
            }

            const double f1 = 2 * (precision * sensitivity) / (precision + sensitivity);

            if (!std::isnan(f1)) {
                sumF1 += f1;
                f1Count++;
            }

            result.labels.emplace_back(label.getName(), accuracy, sensitivity, precision, f1, cnts);
        }

        result.globalAccuracy = static_cast<double>(sumTp) / totalSamples;

        result.avgAccuracy = sumAccuracy / accuracyCount;
        result.avgF1 = sumF1 / f1Count;
        result.avgPrecision = sumPrecision / precisionCount;
        result.avgSensitivity = sumSensitivity / sensitivityCount;

        return result;
    }


};




#endif