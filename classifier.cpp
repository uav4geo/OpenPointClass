#include "classifier.hpp"

Regularization parseRegularization(const std::string &regularization){
    if (regularization == "none") return None;
    else if (regularization == "local_smooth") return LocalSmooth;
    else throw std::runtime_error("Invalid regularization value: " + regularization);
}



