#ifndef _CGAL_H
#define _CGAL_H

#include <CGAL/Simple_cartesian.h>
#include <CGAL/Classification.h>
#include <CGAL/Point_set_3.h>

typedef CGAL::Simple_cartesian<double> Kernel;
typedef Kernel::Point_3 Point;
typedef std::vector<Point> Point_range;
typedef CGAL::Point_set_3<Point> Point_set;
typedef std::array<uint8_t, 3> Color;

typedef Point_set::Point_map Pmap;
typedef Point_set::Property_map<int> Imap;
typedef Point_set::Property_map<uint8_t> UCmap;
typedef Point_set::Property_map<Color> Color_map;

namespace Classification = CGAL::Classification;
typedef Classification::Label_handle Label_handle;
typedef Classification::Feature_handle Feature_handle;
typedef Classification::Label_set Label_set;
typedef Classification::Feature_set Feature_set;
typedef Classification::ETHZ::Random_forest_classifier Classifier;
typedef Classification::Point_set_feature_generator<Kernel, Point_set, Pmap>    Feature_generator;
typedef Classification::Local_eigen_analysis Local_eigen_analysis;

#endif