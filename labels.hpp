#ifndef LABELS_H
#define LABELS_H

#include <vector>
#include <string>
#include <unordered_map>
#include "color.hpp"

#define LABEL_UNASSIGNED 255
#define LABEL_UNCLASSIFIED 0

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
  int getAsprsCode() const { return asprsCode; }
  Color getColor() const { return color; }
};

std::vector<Label> getLabels();
std::vector<Label> getTrainingLabels();
std::unordered_map<std::string, int> getTrainingCodes();
std::unordered_map<int, int> getAsprs2TrainCodes();
std::unordered_map<int, int> getTrain2AsprsCodes();

#endif