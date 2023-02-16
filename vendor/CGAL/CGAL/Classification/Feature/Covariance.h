// Author: Piero Toffanin

#ifndef CGAL_CLASSIFICATION_FEATURE_COVARIANCE_H
#define CGAL_CLASSIFICATION_FEATURE_COVARIANCE_H

#include <CGAL/license/Classification.h>

#include <vector>
#include <CGAL/Classification/Feature_base.h>
#include <CGAL/Classification/Local_eigen_analysis.h>

namespace CGAL {

namespace Classification {

namespace Feature {

class SurfaceVariation : public Feature_base
{
  const Classification::Local_eigen_analysis& eigen;
public:
  SurfaceVariation (const Classification::Local_eigen_analysis& eigen)
    : eigen(eigen){
    this->set_name ("surface_variation");
  }
  virtual float value (std::size_t pt_index){
    return eigen.eigenvalue(pt_index)[0];
  }
};

class Omnivariance : public Feature_base
{
  const Classification::Local_eigen_analysis& eigen;
public:
  Omnivariance (const Classification::Local_eigen_analysis& eigen)
    : eigen(eigen){
    this->set_name ("omnivariance");
  }
  virtual float value (std::size_t pt_index){
    return CGAL::kth_root<float>(3, eigen.eigenvalue(pt_index)[0] *
            eigen.eigenvalue(pt_index)[1] *
            eigen.eigenvalue(pt_index)[2]);
  }
};

class Eigenentropy : public Feature_base
{
  const Classification::Local_eigen_analysis& eigen;
public:
  Eigenentropy (const Classification::Local_eigen_analysis& eigen)
    : eigen(eigen){
    this->set_name ("eigenentropy");
  }
  virtual float value (std::size_t pt_index){
    float sum = 0;
    for (std::size_t i = 0; i < 3; i++){
        float l = eigen.eigenvalue(pt_index)[i];
        sum += l * std::log(l);
    }
    return -sum;
  }
};

class Anisotropy : public Feature_base
{
  const Classification::Local_eigen_analysis& eigen;
public:
  Anisotropy (const Classification::Local_eigen_analysis& eigen)
    : eigen(eigen){
    this->set_name ("anisotropy");
  }
  virtual float value (std::size_t pt_index){
    return (eigen.eigenvalue(pt_index)[2] - 
            eigen.eigenvalue(pt_index)[0]) / eigen.eigenvalue(pt_index)[2];
  }
};

class Planarity : public Feature_base
{
  const Classification::Local_eigen_analysis& eigen;
public:
  Planarity (const Classification::Local_eigen_analysis& eigen)
    : eigen(eigen){
    this->set_name ("planarity");
  }
  virtual float value (std::size_t pt_index){
    return (eigen.eigenvalue(pt_index)[1] - 
            eigen.eigenvalue(pt_index)[0]) / eigen.eigenvalue(pt_index)[2];
  }
};

class Linearity : public Feature_base
{
  const Classification::Local_eigen_analysis& eigen;
public:
  Linearity (const Classification::Local_eigen_analysis& eigen)
    : eigen(eigen){
    this->set_name ("linearity");
  }
  virtual float value (std::size_t pt_index){
    return (eigen.eigenvalue(pt_index)[2] - 
            eigen.eigenvalue(pt_index)[1]) / eigen.eigenvalue(pt_index)[2];
  }
};

class Scatter : public Feature_base
{
  const Classification::Local_eigen_analysis& eigen;
public:
  Scatter (const Classification::Local_eigen_analysis& eigen)
    : eigen(eigen){
    this->set_name ("scatter");
  }
  virtual float value (std::size_t pt_index){
    return eigen.eigenvalue(pt_index)[0] / eigen.eigenvalue(pt_index)[2];
  }
};

} // namespace Feature

} // namespace Classification

} // namespace CGAL

#endif // CGAL_CLASSIFICATION_FEATURE_COVARIANCE_H
