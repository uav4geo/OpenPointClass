
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

class Stat
{
public:
    double overallAccuracy;
    std::vector<double> labelsAccuracy;
    double avgIou;
    std::vector<double> f1Scores;
    double avgF1;
};


class Counts
{


public:

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



class Statistics
{

    std::map<int, Counts> stats;
    std::vector<int> labels;
    int totalSamples = 0;

public:

    Statistics(const std::vector<int> &labels)
    {
        for (const auto label : labels)
        {
            stats[label] = Counts();
        }

        // Copy label vector
        this->labels = labels;
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


    std::vector<std::vector<double>> getConfusionMatrix() const
    {
        std::vector<std::vector<double>> result;
        result.resize(labels.size());

        for (const auto label : labels)
        {
            result[label].resize(labels.size());
        }

        for (const auto &[label, ct] : stats)
        {
            const double tp = ct.tp;
            const double fp = ct.fp;
            const double fn = ct.fn;

            const double sensitivity = tp / (tp + fn);

            result[label][label] = sensitivity;

            for (const auto label2 : labels)
            {
                if (label == label2)
                {
                    continue;
                }

                const double tp2 = stats.at(label2).tp;
                const double fn2 = stats.at(label2).fn;

                const double sensitivity2 = tp2 / (tp2 + fn2);

                result[label][label2] = sensitivity2;
            }

        }

        return result;
    }

    Stat getStatistics()
    {
        Stat result;

        std::vector<double> labelsAccuracy(labels.size());
        std::vector<double> f1Scores(labels.size());
        const auto labelsCount = static_cast<double>(labels.size());

        int sumTp = 0;
        double sumIou = 0.0;
        double sumF1 = 0.0;

        for (size_t i = 0; i < labels.size(); ++i)
        {
            int label = labels[i];
            const Counts &cnts = stats[label];
            sumTp += cnts.tp;

            const double accuracy = static_cast<double>(cnts.tp) / (cnts.tp + cnts.fn + cnts.fp);
            labelsAccuracy[i] = accuracy;
              
            // Skip if nan
            if (!isnan(accuracy))
                sumIou += accuracy;

            const double precision = static_cast<double>(cnts.tp) / (cnts.tp + cnts.fp);
            const double sensitivity = static_cast<double>(cnts.tp) / (cnts.tp + cnts.fn);
            const double f1 = 2 * (precision * sensitivity) / (precision + sensitivity);
            f1Scores[i] = f1;

            if (!isnan(f1))
                sumF1 += f1;
        }

        const double overallAccuracy = static_cast<double>(sumTp) / totalSamples;
        const double MeanIou = sumIou / labelsCount;
        const double meanF1Score = sumF1 / labelsCount;

        result.overallAccuracy = overallAccuracy;
        result.labelsAccuracy = labelsAccuracy;
        result.avgIou = MeanIou;
        result.f1Scores = f1Scores;
        result.avgF1 = meanF1Score;

        return result;
    }


};




#endif