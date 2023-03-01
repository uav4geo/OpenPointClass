#include <array>
#include "color.hpp"

std::array<float, 3> rgb2hsv(double r, double g, double b){
    r /= 255.;
    g /= 255.;
    b /= 255.;
    
    double color_max = (std::max) (r, (std::max) (g, b));
    double color_min = (std::min) (r, (std::min) (g, b));
    double diff = color_max - color_min;
    double hue = 0.;

    if (diff > 0.){
      if (color_max == r) hue = 60. * ((g - b) / diff);
      else if (color_max == g) hue = 60. * (((b - r) / diff) + 2.);
      else hue = 60. * (((r - g) / diff) + 4.);
    }

    if (hue < 0.) hue += 360.;

    double saturation = (color_max == 0. ? 0. : 100. * (diff / color_max));
    double value = 100. * color_max;

    return { static_cast<float>(hue), static_cast<float>(saturation), static_cast<float>(value) };
}
