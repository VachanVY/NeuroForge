/*
The style of writing C++ used here
    * no using fancy C++ things
    * mainly going to write C++ like C with Classes with some minor C++ things
    * Use cout instead of printf
*/

#pragma once
// basic cpp libs
#include <iostream>
#include <cstring>
#include <cstdlib>


template <typename dtype>
dtype* copy(dtype* arr, int len){
    dtype* new_arr = (dtype*)std::malloc(sizeof(dtype)*len);
    std::memcpy(new_arr, arr, len*sizeof(dtype));
    return new_arr;
}
