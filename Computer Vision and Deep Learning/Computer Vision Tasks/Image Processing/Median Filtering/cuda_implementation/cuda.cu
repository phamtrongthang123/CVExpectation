#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(error) << std::endl; \
            exit(1); \
        } \
    } while(0)

__device__ void bubbleSort(float* arr, int n) {
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                float temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
}

__device__ float quickSelect(float* arr, int n) {
    // Simple bubble sort for small arrays (kernel size typically <= 25)
    bubbleSort(arr, n);
    return arr[n / 2];
}

__global__ void medianFilterKernel(float* input, float* output, int width, int height, int kernelSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int pad = kernelSize / 2;
    int windowSize = kernelSize * kernelSize;
    float window[25]; // Maximum 5x5 kernel
    
    int idx = 0;
    for (int ky = -pad; ky <= pad; ky++) {
        for (int kx = -pad; kx <= pad; kx++) {
            int nx = min(max(x + kx, 0), width - 1);
            int ny = min(max(y + ky, 0), height - 1);
            window[idx++] = input[ny * width + nx];
        }
    }
    
    output[y * width + x] = quickSelect(window, windowSize);
}

__global__ void medianFilterSharedKernel(float* input, float* output, int width, int height, int kernelSize) {
    extern __shared__ float sharedMem[];
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    
    int pad = kernelSize / 2;
    int sharedWidth = blockDim.x + 2 * pad;
    int sharedHeight = blockDim.y + 2 * pad;
    
    // Load data into shared memory
    for (int i = tid; i < sharedWidth * sharedHeight; i += blockDim.x * blockDim.y) {
        int sy = i / sharedWidth;
        int sx = i % sharedWidth;
        int gx = blockIdx.x * blockDim.x + sx - pad;
        int gy = blockIdx.y * blockDim.y + sy - pad;
        
        gx = min(max(gx, 0), width - 1);
        gy = min(max(gy, 0), height - 1);
        
        sharedMem[i] = input[gy * width + gx];
    }
    
    __syncthreads();
    
    if (x >= width || y >= height) return;
    
    int windowSize = kernelSize * kernelSize;
    float window[25];
    
    int idx = 0;
    for (int ky = 0; ky < kernelSize; ky++) {
        for (int kx = 0; kx < kernelSize; kx++) {
            int sx = threadIdx.x + kx;
            int sy = threadIdx.y + ky;
            window[idx++] = sharedMem[sy * sharedWidth + sx];
        }
    }
    
    output[y * width + x] = quickSelect(window, windowSize);
}

class MedianFilterCUDA {
private:
    float* d_input;
    float* d_output;
    int width, height;
    
public:
    MedianFilterCUDA(int w, int h) : width(w), height(h) {
        CHECK_CUDA(cudaMalloc(&d_input, width * height * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_output, width * height * sizeof(float)));
    }
    
    ~MedianFilterCUDA() {
        cudaFree(d_input);
        cudaFree(d_output);
    }
    
    cv::Mat filter(const cv::Mat& input, int kernelSize, bool useSharedMemory = false) {
        // Copy input to GPU
        CHECK_CUDA(cudaMemcpy(d_input, input.ptr<float>(), 
                             width * height * sizeof(float), cudaMemcpyHostToDevice));
        
        // Configure grid and block dimensions
        dim3 blockSize(16, 16);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                      (height + blockSize.y - 1) / blockSize.y);
        
        if (useSharedMemory) {
            int pad = kernelSize / 2;
            int sharedSize = (blockSize.x + 2 * pad) * (blockSize.y + 2 * pad) * sizeof(float);
            medianFilterSharedKernel<<<gridSize, blockSize, sharedSize>>>(
                d_input, d_output, width, height, kernelSize);
        } else {
            medianFilterKernel<<<gridSize, blockSize>>>(
                d_input, d_output, width, height, kernelSize);
        }
        
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        
        // Copy result back to CPU
        cv::Mat output(height, width, CV_32F);
        CHECK_CUDA(cudaMemcpy(output.ptr<float>(), d_output,
                             width * height * sizeof(float), cudaMemcpyDeviceToHost));
        
        return output;
    }
};

int main() {
    // Initialize CUDA
    int deviceCount;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        std::cerr << "No CUDA capable devices found!" << std::endl;
        return -1;
    }
    
    std::cout << "Found " << deviceCount << " CUDA device(s)" << std::endl;
    
    // Load Lena image
    cv::Mat original_8u = cv::imread("lena.jpg", cv::IMREAD_GRAYSCALE);
    if (original_8u.empty()) {
        std::cout << "Could not load lena.jpg, creating synthetic image" << std::endl;
        original_8u = cv::Mat::zeros(800, 600, CV_8U);
        cv::randu(original_8u, 0, 255);
    } else {
        std::cout << "Loaded lena.jpg: " << original_8u.size() << std::endl;
    }
    
    // Convert to float
    cv::Mat original;
    original_8u.convertTo(original, CV_32F);
    
    int width = original.cols;
    int height = original.rows;
    
    // Add salt and pepper noise
    cv::Mat noisy = original.clone();
    cv::Mat noise_mask(original.size(), CV_8U);
    cv::randu(noise_mask, 0, 100);
    
    int noise_pixels = 0;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (noise_mask.at<uchar>(i, j) < 8) { // 8% noise
                if (rand() % 2) {
                    noisy.at<float>(i, j) = 255.0f; // Salt
                } else {
                    noisy.at<float>(i, j) = 0.0f;   // Pepper
                }
                noise_pixels++;
            }
        }
    }
    
    std::cout << "Added " << noise_pixels << " noise pixels ("
              << (100.0f * noise_pixels) / (height * width) << "%)" << std::endl;
    
    MedianFilterCUDA filter(width, height);
    int kernelSize = 5;
    
    std::cout << "\nApplying CUDA median filter..." << std::endl;
    
    // Test basic CUDA implementation
    auto start = std::chrono::high_resolution_clock::now();
    cv::Mat filtered_basic = filter.filter(noisy, kernelSize, false);
    auto end = std::chrono::high_resolution_clock::now();
    auto basic_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Test shared memory implementation
    start = std::chrono::high_resolution_clock::now();
    cv::Mat filtered_shared = filter.filter(noisy, kernelSize, true);
    end = std::chrono::high_resolution_clock::now();
    auto shared_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Compare with OpenCV CPU implementation
    start = std::chrono::high_resolution_clock::now();
    cv::Mat filtered_opencv;
    cv::medianBlur(noisy, filtered_opencv, kernelSize);
    end = std::chrono::high_resolution_clock::now();
    auto opencv_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Performance Results:" << std::endl;
    std::cout << "  CUDA Basic: " << basic_time.count() << "ms" << std::endl;
    std::cout << "  CUDA Shared Memory: " << shared_time.count() << "ms" << std::endl;
    std::cout << "  OpenCV CPU: " << opencv_time.count() << "ms" << std::endl;
    std::cout << "  Speedup (CPU/CUDA shared): " << (float)opencv_time.count() / shared_time.count() << "x" << std::endl;
    
    // Save results
    cv::Mat display_original, display_noisy, display_filtered;
    original.convertTo(display_original, CV_8U);
    noisy.convertTo(display_noisy, CV_8U);
    filtered_shared.convertTo(display_filtered, CV_8U);
    
    cv::imwrite("lena_original_cuda.png", display_original);
    cv::imwrite("lena_noisy_cuda.png", display_noisy);
    cv::imwrite("lena_filtered_cuda.png", display_filtered);
    
    std::cout << "Images saved: lena_original_cuda.png, lena_noisy_cuda.png, lena_filtered_cuda.png" << std::endl;
    
    return 0;
}
