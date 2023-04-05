#ifndef COLOR_H
#define COLOR_H

#include <cstdint>

std::array<float, 3> rgb2hsv(double r, double g, double b);

struct Color {
    uint8_t r, g, b;
    Color() : r(255), g(255), b(255) {};
    Color(uint8_t r, uint8_t g, uint8_t b) :
        r(r), g(g), b(b) {};
};

#endif