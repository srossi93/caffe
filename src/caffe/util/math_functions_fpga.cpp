#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {
    
    template <typename Dtype>
    void caffe_fpga_add(const int N, const Dtype* a, const Dtype* b, Dtype* y){
        int i = 0;
        for(i = 0; i < N; i++){
            y[i] = a[i] + b[i];
        }
    }

    template void caffe_fpga_add<int>(const int N, const int* a, const int* b, int* y);
    template void caffe_fpga_add<float>(const int N, const float* a, const float* b, float* y);
    template void caffe_fpga_add<double>(const int N, const double* a, const double* b, double* y);


    template <typename Dtype>
    void caffe_fpga_sub(const int N, const Dtype* a, const Dtype* b, Dtype* y){
        int i = 0;
        for(i = 0; i < N; i++){
            y[i] = a[i] - b[i];
        }
    }

    template void caffe_fpga_sub<int>(const int N, const int* a, const int* b, int* y);
    template void caffe_fpga_sub<float>(const int N, const float* a, const float* b, float* y);
    template void caffe_fpga_sub<double>(const int N, const double* a, const double* b, double* y);

}
