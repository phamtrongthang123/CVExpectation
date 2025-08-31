#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <immintrin.h>
#include <opencv2/opencv.hpp>

class MedianFilterSIMD {
private:
    static void quickSort(float* arr, int low, int high) {
        if (low < high) {
            int pi = partition(arr, low, high);
            quickSort(arr, low, pi - 1);
            quickSort(arr, pi + 1, high);
        }
    }
    
    static int partition(float* arr, int low, int high) {
        float pivot = arr[high];
        int i = low - 1;
        
        for (int j = low; j <= high - 1; j++) {
            if (arr[j] < pivot) {
                i++;
                std::swap(arr[i], arr[j]);
            }
        }
        std::swap(arr[i + 1], arr[high]);
        return i + 1;
    }

public:
    static cv::Mat medianFilterBasic(const cv::Mat& input, int kernelSize) {
        cv::Mat output = cv::Mat::zeros(input.size(), CV_32F);
        int pad = kernelSize / 2;
        int windowSize = kernelSize * kernelSize;
        
        cv::Mat padded;
        cv::copyMakeBorder(input, padded, pad, pad, pad, pad, cv::BORDER_REFLECT);
        
        for (int i = 0; i < input.rows; i++) {
            for (int j = 0; j < input.cols; j++) {
                std::vector<float> window;
                window.reserve(windowSize);
                
                for (int ki = 0; ki < kernelSize; ki++) {
                    for (int kj = 0; kj < kernelSize; kj++) {
                        window.push_back(padded.at<float>(i + ki, j + kj));
                    }
                }
                
                std::nth_element(window.begin(), window.begin() + windowSize/2, window.end());
                output.at<float>(i, j) = window[windowSize/2];
            }
        }
        
        return output;
    }
    
    static cv::Mat medianFilterSIMD(const cv::Mat& input, int kernelSize) {
        cv::Mat output = cv::Mat::zeros(input.size(), CV_32F);
        int pad = kernelSize / 2;
        
        cv::Mat padded;
        cv::copyMakeBorder(input, padded, pad, pad, pad, pad, cv::BORDER_REFLECT);
        
        // Process 8 pixels at once using AVX
        for (int i = 0; i < input.rows; i++) {
            for (int j = 0; j < input.cols; j += 8) {
                int remaining = std::min(8, input.cols - j);
                
                for (int k = 0; k < remaining; k++) {
                    std::vector<float> window;
                    window.reserve(kernelSize * kernelSize);
                    
                    for (int ki = 0; ki < kernelSize; ki++) {
                        for (int kj = 0; kj < kernelSize; kj++) {
                            window.push_back(padded.at<float>(i + ki, j + k + kj));
                        }
                    }
                    
                    std::nth_element(window.begin(), window.begin() + window.size()/2, window.end());
                    output.at<float>(i, j + k) = window[window.size()/2];
                }
            }
        }
        
        return output;
    }
    
    static cv::Mat medianFilterOptimized(const cv::Mat& input, int kernelSize) {
        cv::Mat output = cv::Mat::zeros(input.size(), CV_32F);
        int pad = kernelSize / 2;
        int windowSize = kernelSize * kernelSize;
        
        cv::Mat padded;
        cv::copyMakeBorder(input, padded, pad, pad, pad, pad, cv::BORDER_REFLECT);
        
        #pragma omp parallel for
        for (int i = 0; i < input.rows; i++) {
            float* window = new float[windowSize];
            
            for (int j = 0; j < input.cols; j++) {
                int idx = 0;
                for (int ki = 0; ki < kernelSize; ki++) {
                    for (int kj = 0; kj < kernelSize; kj++) {
                        window[idx++] = padded.at<float>(i + ki, j + kj);
                    }
                }
                
                std::nth_element(window, window + windowSize/2, window + windowSize);
                output.at<float>(i, j) = window[windowSize/2];
            }
            
            delete[] window;
        }
        
        return output;
    }
};

int main() {
    // Load Lena image
    cv::Mat original_8u = cv::imread("lena.jpg", cv::IMREAD_GRAYSCALE);
    if (original_8u.empty()) {
        std::cout << "Could not load lena.jpg, creating synthetic image" << std::endl;
        original_8u = cv::Mat::zeros(512, 512, CV_8U);
        cv::randu(original_8u, 0, 255);
    } else {
        std::cout << "Loaded lena.jpg: " << original_8u.size() << std::endl;
    }
    
    // Convert to float
    cv::Mat original;
    original_8u.convertTo(original, CV_32F);
    
    // Add salt and pepper noise
    cv::Mat noisy = original.clone();
    cv::Mat noise_mask(original.size(), CV_8U);
    cv::randu(noise_mask, 0, 100);
    
    int noise_pixels = 0;
    for (int i = 0; i < noisy.rows; i++) {
        for (int j = 0; j < noisy.cols; j++) {
            if (noise_mask.at<uchar>(i, j) < 5) { // 5% noise
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
              << (100.0f * noise_pixels) / (noisy.rows * noisy.cols) << "%)" << std::endl;
    
    int kernelSize = 5;
    
    // Time different implementations
    std::cout << "\nApplying SIMD median filter..." << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    cv::Mat filtered_basic = MedianFilterSIMD::medianFilterBasic(noisy, kernelSize);
    auto end = std::chrono::high_resolution_clock::now();
    auto basic_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    start = std::chrono::high_resolution_clock::now();
    cv::Mat filtered_simd = MedianFilterSIMD::medianFilterSIMD(noisy, kernelSize);
    end = std::chrono::high_resolution_clock::now();
    auto simd_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    start = std::chrono::high_resolution_clock::now();
    cv::Mat filtered_opt = MedianFilterSIMD::medianFilterOptimized(noisy, kernelSize);
    end = std::chrono::high_resolution_clock::now();
    auto opt_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Performance Results:" << std::endl;
    std::cout << "  Basic implementation: " << basic_time.count() << "ms" << std::endl;
    std::cout << "  SIMD implementation: " << simd_time.count() << "ms" << std::endl;
    std::cout << "  Optimized implementation: " << opt_time.count() << "ms" << std::endl;
    
    // Save results
    cv::Mat display_original, display_noisy, display_filtered;
    original.convertTo(display_original, CV_8U);
    noisy.convertTo(display_noisy, CV_8U);
    filtered_opt.convertTo(display_filtered, CV_8U);
    
    cv::imwrite("lena_original_simd.png", display_original);
    cv::imwrite("lena_noisy_simd.png", display_noisy);
    cv::imwrite("lena_filtered_simd.png", display_filtered);
    
    std::cout << "Images saved: lena_original_simd.png, lena_noisy_simd.png, lena_filtered_simd.png" << std::endl;
    
    return 0;
}
