// Author: Piero Toffanin

#ifndef CGAL_CLASSIFICATION_FEATURE_MOMENTS_H
#define CGAL_CLASSIFICATION_FEATURE_MOMENTS_H

#include <CGAL/license/Classification.h>

#include <vector>
#include <CGAL/Classification/Feature_base.h>
#include <CGAL/Classification/Local_eigen_analysis.h>
#include <CGAL/Classification/Image.h>
#include <CGAL/Classification/Planimetric_grid.h>

namespace CGAL {

namespace Classification {

namespace Feature {

template <typename GeomTraits, typename PointRange, typename PointMap>
class OrderAxis : public Feature_base
{
  using Image_float = Image<float>;
  
  int order;
  std::size_t axis;
  
  typedef Planimetric_grid<GeomTraits, PointRange, PointMap> Grid;

  const PointRange& input;
  PointMap point_map;
  const Grid& grid;
  Image_float dtm;
  std::vector<float> values;
 
  const Classification::Local_eigen_analysis& eigen;
public:

  OrderAxis (int order, std::size_t axis, 
                const PointRange& input,
                PointMap point_map,
                const Grid& grid, 
                const Classification::Local_eigen_analysis& eigen)
    : order(order), axis(axis), input(input), point_map(point_map), grid(grid), eigen(eigen){
    std::stringstream ss;
    ss << "order_" << order << "_" << axis;
    this->set_name (ss.str());

    dtm = Image_float(grid.width(),grid.height());

    for (std::size_t j = 0; j < grid.height(); ++ j)
      for (std::size_t i = 0; i < grid.width(); ++ i)
        if (grid.has_points(i,j)){
          float sum = 0.f;

          typename Grid::iterator end = grid.indices_end(i,j);
          for (typename Grid::iterator it = grid.indices_begin(i,j); it != end; ++ it){
            typename GeomTraits::Point_3 centroid = eigen.centroid<GeomTraits>(*it);
            typename GeomTraits::Vector_3 e = eigen.eigenvector<GeomTraits>(*it, (2 - (axis - 1)));
            typename GeomTraits::Point_3 p = get(point_map, *(input.begin()+(*it)));
            float v = (p - centroid) * e;
            for (int k = 1; k < order; k++) v *= v;
            sum += v;
          }

          dtm(i,j) = sum;
        }

    // for (std::size_t i = 0; i < input.size(); i++){
    //   float sum = 0.f;
    //   typename GeomTraits::Point_3 centroid = eigen.centroid<GeomTraits>(i);
    //   typename GeomTraits::Vector_3 e = eigen.eigenvector<GeomTraits>(i, (2 - (axis - 1)));

    //   std::size_t I = grid.x(i);
    //   std::size_t J = grid.y(i);
    //   typename Grid::iterator end = grid.indices_end(I,J);

    //   for (typename Grid::iterator it = grid.indices_begin(I,J); it != end; ++ it){
    //     typename GeomTraits::Point_3 p = get(point_map, *(input.begin()+(*it)));
    //     float v = (p - centroid) * e;
    //     for (int k = 1; k < order; k++) v *= v;
        
    //     sum += v;
    //   }

    //}
  }
  virtual float value (std::size_t pt_index){
    //return values[pt_index];
    std::size_t I = grid.x(pt_index);
    std::size_t J = grid.y(pt_index);
    return dtm(I,J);
  }
};

} // namespace Feature

} // namespace Classification

} // namespace CGAL

#endif // CGAL_CLASSIFICATION_FEATURE_MOMENTS_H
