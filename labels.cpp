#include "labels.hpp"

std::vector<Label> getLabels() {
    std::vector<Label> l;

    l.emplace_back("unassigned", LABEL_UNASSIGNED, 1);
    l.emplace_back("unclassified", 0, 1, Color(255, 255, 255));

    l.emplace_back("ground", 1, 2, Color(192, 192, 192));
    l.emplace_back("low_vegetation", 2, 3, Color(0, 128, 0));
    l.emplace_back("medium_vegetation", 3, 4, Color(0, 128, 0));
    l.emplace_back("high_vegetation", 4, 5, Color(0, 128, 0));
    l.emplace_back("building", 5, 6, Color(255, 126, 0));
    l.emplace_back("low_point", 6, 7, Color(100, 100, 100));

    l.emplace_back("water", 7, 9, Color(0, 128, 255));
    l.emplace_back("rail", 8, 10, Color(150, 70, 70));

    l.emplace_back("road_surface", 9, 11, Color(110, 110, 110));
    l.emplace_back("wire_guard", 10, 13, Color(135, 135, 135));
    l.emplace_back("wire_conductor", 11, 14, Color(255, 255, 255));
    l.emplace_back("transmission_tower", 12, 15, Color(90, 90, 90));
    l.emplace_back("wire_structure_connector", 13, 16, Color(110, 150, 160));
    l.emplace_back("bridge_deck", 14, 17, Color(190, 190, 190));
    l.emplace_back("high_noise", 15, 18, Color(255, 0, 0));
    l.emplace_back("overhead_structure", 16, 19, Color(78, 78, 78));
    l.emplace_back("ignored_ground", 17, 20, Color(180, 180, 180));
    l.emplace_back("snow", 18, 21, Color(207, 207, 207));
    l.emplace_back("temporal_exclusion", 19, 22, Color(128, 220, 128));

    l.emplace_back("human_made_object", 20, 64, Color(255, 255, 0));


    return l;
}
std::vector<Label> getTrainingLabels() {
    const auto labels = getLabels();
    std::vector<Label> out;
    for (auto &l : labels) {
        if (l.getTrainingCode() != LABEL_UNASSIGNED) {
            out.push_back(l);
        }
    }
    return out;
}

std::unordered_map<std::string, int> getTrainingCodes() {
    std::unordered_map<std::string, int> out;

    const auto labels = getLabels();
    for (const auto &l : labels) {
        out[l.getName()] = l.getTrainingCode();
    }

    return out;
}

std::unordered_map<int, int> getAsprs2TrainCodes() {
    std::unordered_map<int, int> out;

    const auto labels = getLabels();
    for (const auto &l : labels) {
        out[l.getAsprsCode()] = l.getTrainingCode();
    }

    for (int i = 0; i < 255; i++) {
        if (out.find(i) == out.end()) out[i] = LABEL_UNASSIGNED;
    }

    return out;
}

std::unordered_map<int, int> getTrain2AsprsCodes() {
    std::unordered_map<int, int> out;

    const auto labels = getLabels();
    for (const auto &l : labels) {
        out[l.getTrainingCode()] = l.getAsprsCode();
    }

    for (int i = 0; i < 255; i++) {
        if (out.find(i) == out.end()) out[i] = 1;
    }

    return out;
}