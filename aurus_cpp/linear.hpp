#pragma once
#include <iostream>
#include <cstring>

#include "utils.hpp"
#include "linear.hpp"


template <typename dtype>
void matmul(  // (A, B) @ (B, C) = (A, C)
    dtype* out,        // (A, C)
    const dtype* in1,  // (A, B)
    const dtype* in2,  // (B, C)
    const dtype* init, // (C,)
    int A, int B, int C
){
    for(int a=0; a<A; a++){
        for(int c=0; c<C; c++){
            out[a*C + c] = (init != NULL) ? init[c] : ((dtype)0);
            for(int b=0; b<B; b++){
                out[a*C + c] += in1[a*B + b]*in2[b*C + c];
            }
        }
    }
}


template <typename dtype>
dtype* transpose(
    dtype* a,  // (p, q)
    int p, int q
){
    dtype* aT = copy<dtype>(a, p*q);
    for(int i=0; i<p; i++){
        for(int j=0; j<q; j++){
            aT[p*j+i] = a[q*i+j];
        }
    }
    return aT;
}


template <typename dtype>
void linear_forward(
    dtype* out,  // (B, fan_out)
    dtype* x,    // (B, fan_in)
    dtype* wie,  // (fan_in, fan_out)
    dtype* bias, // (fan_out,)
    int B, int fin, int fout
){
    matmul<dtype>(out, x, wie, bias, B, fin, fout);
}


template <typename dtype>
void linear_backward(
    dtype* dL_dx,       // (B, fan_in)
    dtype* dL_dwie,     // (fan_in, fan_out)
    dtype* dL_db,       // (fan_out,)
    const dtype* dL_dO, // (B, fan_out)
    const dtype* x,     // (B, fan_in)
    const dtype* wie,   // (fan_in, fan_out)
    int B, int fin, int fout
){
    // dL_dx = dL_dO @ wie.T | (B, fan_in) <= (B, fan_out) @ (fan_in, fan_out).T
    matmul<dtype>(dL_dx, dL_dO, wie, NULL, B, fout, fin);

    // dL_dwie = x.T @ dL_dO | (fan_in, fan_out) <= (B, fan_in).T @ (B, fan_out)
    dtype* xT = transpose<dtype>(x, B, fin);
    matmul<dtype>(dL_dwie, xT, dL_dO, NULL, fin, B, fout);
    free(xT);

    // dL_db = dL_dO.sum(dim=0) | (fan_out,) <= (B, fan_out)
    for(int fo=0; fo<fout; fo++){
        dL_db[fo] = (dtype)0;
        for(int b=0; b<B; b++){
            dL_db[fo] += dL_dO[b*fout + fo];
        }
    }
    
}


template <typename dtype>
class Linear{
    public:
        int fan_in;
        int fan_out;
        dtype* wie;
        dtype* bias;


        Linear(int fin, int fout){
            fan_in = fin;
            fan_out = fout;
            wie = (dtype*)malloc(sizeof(dtype)*fin*fout);
            bias = (dtype*)malloc(sizeof(dtype)*fout);
        }

        ~Linear(){
            free(wie);
            free(bias);
        }

        void forward(dtype* out, dtype* x, int B){
            linear_forward<dtype>(out, x, wie, bias, B, fan_in, fan_out);
        }

        void backward(dtype* dL_dx, dtype* dL_dwie, dtype* dL_db, dtype* dL_dO, dtype* x, int B){
            linear_backward<dtype>(dL_dx, dL_dwie, dL_db, dL_dO, x, wie, B, fan_in, fan_out);
        }
};

// int main(){
//     // Linear<float> l(3, 2);
//     // float x[3] = {1, 2, 3};
//     // float y[2];
//     // l.forward(y, x, 1);
//     // std::cout << y[0] << " " << y[1] << std::endl;
//     // return 0;
// }
