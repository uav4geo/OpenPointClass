#include "labels.hpp"

std::vector<Label *> getLabels(){
    std::vector<Label *> l;

    l.push_back(new Label("unassigned", LABEL_UNASSIGNED, 1));
    l.push_back(new Label("ground", 0, 2, Color(192, 192, 192)));
    l.push_back(new Label("low_vegetation", 1, 3, Color(0, 128, 0)));
    l.push_back(new Label("medium_vegetation", 2, 4, Color(0, 128, 0)));
    l.push_back(new Label("high_vegetation", 3, 5, Color(0, 128, 0)));
    l.push_back(new Label("building", 4, 6, Color(255, 126, 0)));
    l.push_back(new Label("low_point", 5, 7, Color(100, 100, 100)));
    
    l.push_back(new Label("water", 6, 9, Color(0, 128, 255)));
    l.push_back(new Label("rail", 7, 10, Color(150, 70, 70)));
    
    l.push_back(new Label("road_surface", 8, 11, Color(110, 110, 110)));
    l.push_back(new Label("wire_guard", 9, 13, Color(135, 135, 135)));
    l.push_back(new Label("wire_conductor", 10, 14, Color(255, 255, 255)));
    l.push_back(new Label("transmission_tower", 11, 15, Color(90, 90, 90)));
    l.push_back(new Label("wire_structure_connector", 12, 16, Color(110, 150, 160)));
    l.push_back(new Label("bridge_deck", 13, 17, Color(190, 190, 190)));
    l.push_back(new Label("high_noise", 14, 18, Color(255, 0, 0)));
    l.push_back(new Label("overhead_structure", 15, 19, Color(78, 78, 78)));
    l.push_back(new Label("ignored_ground", 16, 20, Color(180, 180, 180)));
    l.push_back(new Label("snow", 17, 21, Color(207, 207, 207)));
    l.push_back(new Label("temporal_exclusion", 18, 22, Color(128, 220, 128)));

    l.push_back(new Label("vehicle", 19, 64, Color(255, 255, 0)));

    return l;
}
std::vector<Label *> getTrainingLabels(){
    auto labels = getLabels();
    std::vector<Label *> out;
    for (auto &l : labels){
        if (l->getTrainingCode() != LABEL_UNASSIGNED){
            out.push_back(l);
        }
    }
    return out;
}

std::unordered_map<std::string, int> getTrainingCodes(){
    std::unordered_map<std::string, int> out;

    auto labels = getLabels();
    for (auto l : labels){
        out[l->getName()] = l->getTrainingCode();
    }

    return out;
}