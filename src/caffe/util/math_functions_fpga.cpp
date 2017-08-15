#include "caffe/util/math_functions.hpp"

namespace caffe {
#ifdef __SDSCC__
#endif
    template <typename Dtype>
    void caffe_fpga_add(const int N, const Dtype* a, const Dtype* b, Dtype* y){
        int i = 0;
        for(i = 0; i < N; i++){
#ifdef __SDSVHLS__
#pragma HLS UNROLL factor=4
#pragma HLS PIPELINE
#endif
            y[i] = a[i] + b[i];
        }
    }

    template void 
        caffe_fpga_add<int>(const int N, const int* a, const int* b, int* y);
    template void 
        caffe_fpga_add<float>(const int N, const float* a, const float* b, float* y);
    template void 
        caffe_fpga_add<double>(const int N, const double* a, const double* b, double* y);

    template <typename Dtype>
    void caffe_fpga_sub(const int N, const Dtype* a, const Dtype* b, Dtype* y){
        int i = 0;
        for(i = 0; i < N; i++){
            y[i] = a[i] - b[i];
        }
    }

    template void 
        caffe_fpga_sub<int>(const int N, const int* a, const int* b, int* y);
    template void 
        caffe_fpga_sub<float>(const int N, const float* a, const float* b, float* y);
    template void 
        caffe_fpga_sub<double>(const int N, const double* a, const double* b, double* y);

    
    template <typename Dtype>
    void caffe_fpga_mul(const int N, const Dtype* a, const Dtype* b, Dtype* y){
        int i = 0;
        for(i = 0; i < N; i++){
            y[i] = a[i] * b[i];
        }
    }

    template void 
        caffe_fpga_mul<int>(const int N, const int* a, const int* b, int* y);
    template void 
        caffe_fpga_mul<float>(const int N, const float* a, const float* b, float* y);
    template void 
        caffe_fpga_mul<double>(const int N, const double* a, const double* b, double* y);


    template <typename Dtype>
    void caffe_fpga_div(const int N, const Dtype* a, const Dtype* b, Dtype* y){
        int i = 0;
        for(i = 0; i < N; i++){
            if (b[i] != 0)
                y[i] = a[i] / b[i];
            else
                y[i] = 0;
        }
    }

    template void 
        caffe_fpga_div<int>(const int N, const int* a, const int* b, int* y);
    template void 
        caffe_fpga_div<float>(const int N, const float* a, const float* b, float* y);
    template void 
        caffe_fpga_div<double>(const int N, const double* a, const double* b, double* y);
}
