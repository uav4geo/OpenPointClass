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
  
  typedef Image<float> Image_float;
  typedef Planimetric_grid<GeomTraits, PointRange, PointMap> Grid;

  const PointRange& input;
  PointMap point_map;
  const Grid& grid;
  Image_float dtm;
  std::vector<float> values;
 
  const Classification::Local_eigen_analysis& eigen;
public:
    // TODO: does a medoid perform better?
  OrderAxis (int order, std::size_t axis, 
                const PointRange& input,
                PointMap point_map,
                const Grid& grid, 
                const Classification::Local_eigen_analysis& eigen)
    : order(order), axis(axis), input(input), point_map(point_map), grid(grid), eigen(eigen){
    std::stringstream ss;
    ss << "order_" << order << "_" << axis;
    this->set_name (ss.str());
    eigen
    dtm = Image_float(grid.width(),grid.height());

    for (std::size_t j = 0; j < grid.height(); ++ j)
      for (std::size_t i = 0; i < grid.width(); ++ i)
        if (grid.has_points(i,j)){
          float z_min = (std::numeric_limits<float>::max)();

          typename Grid::iterator end = grid.indices_end(i,j);
          for (typename Grid::iterator it = grid.indices_begin(i,j); it != end; ++ it)
          {
            float z = float(get(point_map, *(input.begin()+(*it))).z());
            z_min = ((std::min)(z_min, z));
          }

          dtm(i,j) = z_min;
        }

    if (grid.width() * grid.height() > input.size())
    {
      values.resize (input.size(), 0.f);
      for (std::size_t i = 0; i < input.size(); ++ i)
      {
        std::size_t I = grid.x(i);
        std::size_t J = grid.y(i);
        values[i] = float(get (point_map, *(input.begin() + i)).z() - dtm(I,J));
      }
      dtm.free();
    }
  }
  virtual float value (std::size_t pt_index){
    if (values.empty()){
        std::size_t I = grid.x(pt_index);
        std::size_t J = grid.y(pt_index);
        return float(get (point_map, *(input.begin() + pt_index)).z() - dtm(I,J));
    }

    return values[pt_index];
  }
};

} // namespace Feature

} // namespace Classification

} // namespace CGAL

#endif // CGAL_CLASSIFICATION_FEATURE_MOMENTS_H
