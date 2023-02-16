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
  int order;
  std::size_t axis;
  
  typedef Planimetric_grid<GeomTraits, PointRange, PointMap> Grid;

  const PointRange& input;
  PointMap point_map;
  const Grid& grid;
 
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
    

  }
  virtual float value (std::size_t pt_index){
    // TODO: does a medoid perform better?
    typename GeomTraits::Point_3 centroid = eigen.centroid<GeomTraits>(pt_index);

    std::size_t i = grid.x(pt_index);
    std::size_t j = grid.y(pt_index);
    typename Grid::iterator end = grid.indices_end(i,j);

    float sum = 0.f;
    for (typename Grid::iterator it = grid.indices_begin(i,j); it != end; ++ it){
        typename GeomTraits::Point_3 p = get(point_map, *(input.begin()+(*it)));
        float v = (p - centroid) * eigen.eigenvector<GeomTraits>(pt_index, (2 - (axis - 1)));
        for (int i = 1; i < order; i++){
            v *= v;
        }
        sum += v;
    }

    return sum;
  }
};

} // namespace Feature

} // namespace Classification

} // namespace CGAL

#endif // CGAL_CLASSIFICATION_FEATURE_MOMENTS_H
