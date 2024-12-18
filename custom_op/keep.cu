__global__ void convolution_kernel(const float* A, const float* B, const float* error_matrix, float* C, int M, int N, int K, int batch_size, float act_scale, float act_zp, float filter_scale, float filter_zp){
        //ogni thread si occupa della moltiplicazione di una riga per una colonna
        //identificatori thread
        int row = threadIdx.x + blockIdx.x * BLOCKSIZE;
        int col = threadIdx.y + blockIdx.y * BLOCKSIZE;
        int batch = blockIdx.z;
        //accumulatore delle moltiplicazioni
        long sum = 0;
        long filter_sum = 0;
        long input_sum = 0;
        const float* A_batch = A + batch * M * K;
        float* C_batch = C + batch * M * N;
        if(row < M && col < N && batch < batch_size){
            for(int i=0; i < K; i++){
                float a_value = A_batch[row * K + i];
                float b_value = B[col + i * N];
                filter_sum += b_value * 2;
                input_sum += a_value * 2;
                float error =  error_matrix[(int)a_value * 256 + (int)b_value];
                sum += (a_value * b_value) - error;
            }
            C_batch[row * N + col] += act_scale * filter_scale * (sum - input_sum * filter_zp - filter_sum * act_zp + K * 2 * act_zp * filter_zp);
        }
    }

    torch::Tensor convolution(torch::Tensor A, torch::Tensor B, torch::Tensor error_matrix, double act_scale, double act_zp, double filter_scale, double filter_zp) {
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
            static_cast<float>(act_scale), static_cast<float>(act_zp), static_cast<float>(filter_scale), static_cast<float>(filter_zp)
        );
        return C;

    }
    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}

    TORCH_LIBRARY(custom_op, m) {
        m.def("convolution(Tensor a, Tensor b, Tensor c, float act_scale, float act_zp, float filter_scale, float filter_zp) -> (Tensor)");
        m.def("derivate(Tensor a, Tensor b, Tensor der_matrix, Tensor grad_output) -> (Tensor)");
    }

    TORCH_LIBRARY_IMPL(custom_op, CUDA, m) {
        m.impl("convolution", &convolution);
        m.impl("derivate", &derivate);
    }
}