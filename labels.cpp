#include "labels.hpp"

std::vector<Label> getLabels(){
    std::vector<Label> l;

    l.push_back(Label("unassigned", LABEL_UNASSIGNED, 1));
    l.push_back(Label("ground", 0, 2, Color(128, 64, 0)));
    l.push_back(Label("low_vegetation", 1, 3, Color(0, 128, 0)));
    l.push_back(Label("medium_vegetation", 2, 4, Color(0, 128, 0)));
    l.push_back(Label("high_vegetation", 3, 5, Color(0, 128, 0)));
    l.push_back(Label("building", 4, 6, Color(0, 128, 0)));
    l.push_back(Label("water", 5, 9, Color(0, 128, 255)));
    l.push_back(Label("road_surface", 6, 11, Color(100, 100, 100)));
    l.push_back(Label("vehicle", 7, 64, Color(255, 0, 0)));

    return l;
}
std::vector<Label> getTrainingLabels(){
    auto labels = getLabels();
    std::vector<Label> out;
    for (auto &l : labels){
        if (l.getTrainingCode() != LABEL_UNASSIGNED){
            out.push_back(l);
        }
    }
    return out;
}

std::unordered_map<std::string, int> getTrainingCodes(){
    std::unordered_map<std::string, int> out;

    auto labels = getLabels();
    for (auto l : labels){
        out[l.getName()] = l.getTrainingCode();
    }

    return out;
}