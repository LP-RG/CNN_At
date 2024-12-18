#include <torch/extension.h>
#include <torch/library.h>
#include <curand_kernel.h>
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#define BLOCKSIZE 32
namespace custom_op {

    __global__ void derivate_kernel(const float* A, const float* B, const float* derivate, const float* grad_output,float* A_copy, int M, int N, int K, int batch_size, int out_dim){
        int row = threadIdx.x + blockIdx.x * BLOCKSIZE;
        int col = threadIdx.y + blockIdx.y * BLOCKSIZE;
        int batch = blockIdx.z;
        const float* A_batch = A + batch * M * K;
        float* A_copy_batch = A_copy + batch * N * M * K;
        const float* grad_output_batch = grad_output + batch * N * out_dim * out_dim;
        float grad_value = grad_output_batch[col * out_dim * out_dim + row];
        if(row < M && col < N && batch < batch_size){
            for(int i=0; i < K; i++){
                float a_value = A_batch[row * K + i];
                float b_value = B[col + i * N];
                float value = derivate[std::abs((int)a_value) * 256 + std::abs((int)b_value)];
                float cor_value = value * grad_value;
                A_copy_batch[col * M * K + row * K + i] = cor_value;
            }
        }
    }

    torch::Tensor derivate(torch::Tensor A, torch::Tensor B, torch::Tensor derivate,torch::Tensor grad_output) {
        TORCH_CHECK(A.size(2) == B.size(0), "Le dimensioni non corrispondono per la moltiplicazione!");
        int M = A.size(1);
        int K = A.size(2);
        int N = B.size(1);
        int batch_size = A.size(0);
        int out_dim = std::sqrt(M);
        auto A_copy = at::zeros({batch_size,N, M, K}, torch::TensorOptions(at::kFloat).device(A.device()));
        dim3 threadsPerBlock(BLOCKSIZE, BLOCKSIZE);
        dim3 numBlocks((M + threadsPerBlock.x - 1) / threadsPerBlock.x, (N + threadsPerBlock.y - 1) / threadsPerBlock.y, batch_size);
        derivate_kernel<<<numBlocks, threadsPerBlock>>>(
            A.data_ptr<float>(), B.data_ptr<float>(),derivate.data_ptr<float>(),grad_output.data_ptr<float>(), A_copy.data_ptr<float>(), M, N, K, batch_size,out_dim
        );
        return A_copy;
    }

    __global__ void convolution_kernel(const float* A, const float* B, const float* error_matrix, float* C, int M, int N, int K, int batch_size, float act_scale, float filter_scale){
        //ogni thread si occupa della moltiplicazione di una riga per una colonna
        //identificatori thread
        int row = threadIdx.x + blockIdx.x * BLOCKSIZE;
        int col = threadIdx.y + blockIdx.y * BLOCKSIZE;
        int batch = blockIdx.z;
        //accumulatore delle moltiplicazioni
        long sum = 0;
        const float* A_batch = A + batch * M * K;
        float* C_batch = C + batch * M * N;
        if(row < M && col < N && batch < batch_size){
            for(int i=0; i < K; i++){
                float a_value = A_batch[row * K + i];
                float b_value = B[col + i * N];
                float error =  error_matrix[std::abs((int)a_value) * 256 + std::abs((int)b_value)];
                sum += (a_value * b_value) - error;
            }
            C_batch[row * N + col] += act_scale * filter_scale * sum ;
        }
    }

    torch::Tensor convolution(torch::Tensor A, torch::Tensor B, torch::Tensor error_matrix, double act_scale, double filter_scale) {
        TORCH_CHECK(A.size(2) == B.size(0), "Le dimensioni non corrispondono per la moltiplicazione!");
        int M = A.size(1);
        int K = A.size(2);
        int N = B.size(1);
        int batch_size = A.size(0);
        auto C = at::zeros({batch_size, M, N}, torch::TensorOptions(at::kFloat).device(A.device()));
        dim3 threadsPerBlock(BLOCKSIZE, BLOCKSIZE);
        dim3 numBlocks((M + threadsPerBlock.x - 1) / threadsPerBlock.x, (N + threadsPerBlock.y - 1) / threadsPerBlock.y, batch_size);
        convolution_kernel<<<numBlocks, threadsPerBlock>>>(
            A.data_ptr<float>(), B.data_ptr<float>(),error_matrix.data_ptr<float>(), C.data_ptr<float>(), M, N, K, batch_size, 
            static_cast<float>(act_scale), static_cast<float>(filter_scale)
        );
        return C;

    }

    __global__ void convolution_no_error_kernel(const float* A, const float* B, float* C, int M, int N, int K, int batch_size, float act_scale, float filter_scale){
        //ogni thread si occupa della moltiplicazione di una riga per una colonna
        //identificatori thread
        int row = threadIdx.x + blockIdx.x * BLOCKSIZE;
        int col = threadIdx.y + blockIdx.y * BLOCKSIZE;
        int batch = blockIdx.z;
        //accumulatore delle moltiplicazioni
        long sum = 0;
        const float* A_batch = A + batch * M * K;
        float* C_batch = C + batch * M * N;
        if(row < M && col < N && batch < batch_size){
            for(int i=0; i < K; i++){
                float a_value = A_batch[row * K + i];
                float b_value = B[col + i * N];
                sum += a_value * b_value;
            }
            C_batch[row * N + col] += act_scale * filter_scale * sum ;
        }
    }

    torch::Tensor convolution_no_error(torch::Tensor A, torch::Tensor B, double act_scale, double filter_scale) {
        TORCH_CHECK(A.size(2) == B.size(0), "Le dimensioni non corrispondono per la moltiplicazione!");
        int M = A.size(1);
        int K = A.size(2);
        int N = B.size(1);
        int batch_size = A.size(0);
        auto C = at::zeros({batch_size, M, N}, torch::TensorOptions(at::kFloat).device(A.device()));
        dim3 threadsPerBlock(BLOCKSIZE, BLOCKSIZE);
        dim3 numBlocks((M + threadsPerBlock.x - 1) / threadsPerBlock.x, (N + threadsPerBlock.y - 1) / threadsPerBlock.y, batch_size);
        convolution_no_error_kernel<<<numBlocks, threadsPerBlock>>>(
            A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K, batch_size, 
            static_cast<float>(act_scale), static_cast<float>(filter_scale)
        );
        return C;

    }

    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}

    TORCH_LIBRARY(custom_op, m) {
        m.def("convolution(Tensor a, Tensor b, Tensor c, float act_scale, float filter_scale) -> (Tensor)");
        m.def("convolution_no_error(Tensor a, Tensor b, float act_scale, float filter_scale) -> (Tensor)");
        m.def("derivate(Tensor a, Tensor b, Tensor der_matrix, Tensor grad_output) -> (Tensor)");
    }

    TORCH_LIBRARY_IMPL(custom_op, CUDA, m) {
        m.impl("convolution", &convolution);
        m.impl("convolution_no_error", &convolution_no_error);
        m.impl("derivate", &derivate);
    }
}