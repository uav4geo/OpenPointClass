#ifndef LABELS_H
#define LABELS_H

#include <vector>
#include <string>
#include <unordered_map>
#include "color.hpp"

// static std::unordered_map<std::string, int> asprsCodes = {
//     {"unassigned", 1},
//     {"ground", 2},
//     {"low_vegetation", 3},
//     {"medium_vegetation", 4},
//     {"high_vegetation", 5},
//     {"building", 6},
//     {"noise", 7},
//     {"reserved", 8},
//     {"water", 9},
//     {"rail", 10},
//     {"road_surface", 11},
//     {"reserved_2", 12},
//     {"wire_guard", 13},
//     {"wire_conductor", 14},
//     {"transmission_tower", 15},
//     {"wire_connect", 16},
//     {"bridge_deck", 17},
//     {"high_noise", 18},
//     {"vehicle", 64},
// };

#define LABEL_UNASSIGNED 255

class Label{
private:
 std::string name;
 int trainingCode;
 int asprsCode;
 Color color;
public:
  Label(const std::string &name, int trainingCode, int asprsCode, Color color = Color()) 
    : name(name), trainingCode(trainingCode), asprsCode(asprsCode), color(color) {};
  
  std::string getName() const { return name; } 
  int getTrainingCode() const { return trainingCode; }
  Color getColor() const { return color; }
};

std::vector<Label> getLabels();
std::vector<Label> getTrainingLabels();
std::unordered_map<std::string, int> getTrainingCodes();

#endif