#pragma once

#include <stdint.h>

// arr is an integer type of B bits and bit is in the range [0,B)
#define BIT_GET(arr,bit) (arr & (1uL << bit))
#define BIT_SET(arr,bit) (arr |= (1uL << bit))
#define BIT_UNSET(arr,bit) (arr &= ((-1uL) ^ (1uL << bit)))

