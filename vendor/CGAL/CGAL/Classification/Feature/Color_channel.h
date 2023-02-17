// Copyright (c) 2017 GeometryFactory Sarl (France).
// All rights reserved.
//
// This file is part of CGAL (www.cgal.org).
//
// $URL: https://github.com/CGAL/cgal/blob/v5.5.1/Classification/include/CGAL/Classification/Feature/Color_channel.h $
// $Id: Color_channel.h fb6f703 2021-05-04T14:07:49+02:00 Sébastien Loriot
// SPDX-License-Identifier: GPL-3.0-or-later OR LicenseRef-Commercial
//
// Author(s)     : Simon Giraudot

#ifndef CGAL_CLASSIFICATION_FEATURE_COLOR_CHANNEL_H
#define CGAL_CLASSIFICATION_FEATURE_COLOR_CHANNEL_H

#include <CGAL/license/Classification.h>

#include <vector>

#include <CGAL/Classification/Feature_base.h>
#include <CGAL/array.h>

namespace CGAL {

namespace Classification {

namespace Feature {

  /*!
    \ingroup PkgClassificationFeatures

    %Feature based on HSV colorimetric information. If the input
    point cloud has colorimetric information, it can be used for
    classification purposes.

    The HSV channels are defined this way:

    - Hue ranges from 0 to 360 and measures the general "tint" of the
      color (green, blue, pink, etc.)

    - Saturation ranges from 0 to 100 and measures the "strength" of the
      color (0 is gray and 100 is the fully saturated color)

    - Value ranges from 0 to 100 and measures the "brightness" of the
      color (0 is black and 100 is the fully bright color)

    Its default name is "color_hue", "color_saturation" or
    "color_value", depending on which channel is chosen in the
    constructor.

    \note The user only needs to provide a map to standard (and more common)
    RGB colors, the conversion to HSV is done internally.

    \tparam GeomTraits model of \cgal Kernel.
    \tparam PointRange model of `ConstRange`. Its iterator type
    is `RandomAccessIterator` and its value type is the key type of
    `ColorMap`.
    \tparam ColorMap model of `ReadablePropertyMap` whose key
    type is the value type of the iterator of `PointRange` and value type
    is `CGAL::IO::Color`.
  */
template <typename GeomTraits, typename PointRange, typename ColorMap>
class Color_channel : public Feature_base
{
public:

  /// Selected channel.
  enum Channel
  {
    HUE = 0, ///< 0
    SATURATION = 1, ///< 1
    VALUE = 2 ///< 2
  };

private:

  const PointRange& input;
  ColorMap color_map;
  Channel m_channel;

public:

  /*!
    \brief constructs a feature based on the given color channel.

    \param input point range.
    \param color_map property map to access the colors of the input points.
    \param channel chosen HSV channel.
  */
  Color_channel (const PointRange& input,
                 ColorMap color_map,
                 Channel channel)
    : input(input), color_map(color_map), m_channel (channel)
  {
    if (channel == HUE) this->set_name ("color_hue");
    else if (channel == SATURATION) this->set_name ("color_saturation");
    else if (channel == VALUE) this->set_name ("color_value");
  }

  /// \cond SKIP_IN_MANUAL
  virtual float value (std::size_t pt_index)
  {
    std::array<double, 3> c = get(color_map, *(input.begin()+pt_index)).to_hsv();
    return float(c[std::size_t(m_channel)]);
  }
  /// \endcond
};

template <typename GeomTraits, typename PointRange, typename PointMap, typename ColorMap, typename NeighborQuery>
class Color_channel_neighborhood : public Feature_base
{
public:

  /// Selected channel.
  enum Channel
  {
    HUE = 0, ///< 0
    SATURATION = 1, ///< 1
    VALUE = 2 ///< 2
  };

private:

  const PointRange& input;
  PointMap point_map;
  ColorMap color_map;
  Channel m_channel;
  NeighborQuery neighbor_query;

  std::vector<float> values;
public:

  Color_channel_neighborhood (const PointRange& input,
                 PointMap point_map,
                 ColorMap color_map,
                 Channel channel,
                 const NeighborQuery& neighbor_query)
    : input(input), point_map(point_map), color_map(color_map), m_channel (channel), neighbor_query (neighbor_query)
  {
    if (channel == HUE) this->set_name ("color_hue_neighborhood");
    else if (channel == SATURATION) this->set_name ("color_saturation_neighborhood");
    else if (channel == VALUE) this->set_name ("color_value_neighborhood");

    for (std::size_t i = 0; i < input.size(); i++){
      std::vector<std::size_t> neighbors;
      neighbor_query (get(point_map, *(input.begin()+i)), std::back_inserter (neighbors));

      if (neighbors.size() == 0){
        values.push_back(0.f);
        continue;
      }

      float sum = 0.f;
      for (std::size_t j = 0; j < neighbors.size(); ++ j){
        std::array<double, 3> c = get(color_map, *(input.begin()+neighbors[j])).to_hsv();
        sum += c[std::size_t(m_channel)];
      }
      values.push_back(sum / static_cast<float>(neighbors.size()));
    }
  }

  virtual float value (std::size_t pt_index)
  {
    return values[pt_index];
  }
};

} // namespace Feature

} // namespace Classification

} // namespace CGAL

#endif // CGAL_CLASSIFICATION_FEATURE_COLOR_CHANNEL_H
