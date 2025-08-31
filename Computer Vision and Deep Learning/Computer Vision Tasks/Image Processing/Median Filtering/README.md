Median filter is a nonlinear digital filtering technique, often used to reduce noise in images. Or in signal, it will help reduce noise in general.

Implementations:
- [x] Naive numpy implementation (`naive_numpy.py`)
- [x] Vectorized implementation (`vectorized.py`)
- [x] SIMD implementation (`cpp/simd.cpp`)
- [x] CUDA implementation (`cpp/cuda.cu`)

## Key Functions Explained

### Python
**`np.pad()`** - Adds border padding around image using reflection mode
```python
img = np.array([[1, 2], [3, 4]])  # Input: 2x2
padded = np.pad(img, 1, mode='reflect')  # Output: 4x4
# [[1, 1, 2, 2],
#  [1, 1, 2, 2], 
#  [3, 3, 4, 4],
#  [3, 3, 4, 4]]
```

**`np.lib.stride_tricks.sliding_window_view()`** - Creates sliding windows without copying data
```python
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])  # Input: 3x3
windows = sliding_window_view(arr, (2, 2))  # Output: 2x2x2x2
# Shape: (2, 2, 2, 2) - 4 windows of size 2x2
# windows[0,0] = [[1,2], [4,5]]  # Top-left window
# windows[0,1] = [[2,3], [5,6]]  # Top-right window
```

**`np.median()`** - Computes median value
```python
arr = np.array([1, 3, 2, 5, 4])  # Input
median = np.median(arr)  # Output: 3.0
```

### C++
**`cv::copyMakeBorder()`** - OpenCV padding
```cpp
cv::Mat img = (cv::Mat_<float>(2,2) << 1, 2, 3, 4);  // Input: 2x2
cv::Mat padded;
cv::copyMakeBorder(img, padded, 1, 1, 1, 1, cv::BORDER_REFLECT);  // Output: 4x4
```

**`std::nth_element()`** - Partial sort for median
```cpp
std::vector<float> arr = {1, 3, 2, 5, 4};  // Input
std::nth_element(arr.begin(), arr.begin() + 2, arr.end());  // Partial sort
float median = arr[2];  // Output: 3.0 (median at middle position)
```

**`#pragma omp parallel for`** - CPU parallelization
```cpp
#pragma omp parallel for  // Parallel loop execution
for (int i = 0; i < rows; i++) {
    // Process row i in parallel across CPU cores
}
```

**CUDA `__global__` kernel** - GPU parallel execution
```cpp
__global__ void filterKernel(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // Thread ID
    if (idx < size) output[idx] = process(input[idx]);  // Parallel processing
}
```

## Function Usage Examples
Individual function demonstrations:
```bash
# Python function demos
uv run function_usage_examples/np_pad_demo.py
uv run function_usage_examples/sliding_window_demo.py
uv run function_usage_examples/median_demo.py

# C++ function demos
cd function_usage_examples
make cpp_demo
./cpp_demo
```

## Setup with Pixi (Recommended)

**Install pixi (Windows):**
```bash
winget install prefix-dev.pixi
```

**Setup environment:**
```bash
pixi install
pixi shell
```

**Run tasks:**
```bash
# Python implementations
pixi run run-naive
pixi run run-vectorized

# Function demos
pixi run demo-all

```

## Manual Setup

### Python Usage
Make sure `lena.jpg` is in the current directory, then run:
```bash
uv run naive_numpy.py
uv run vectorized.py
```

All implementations will:
1. Load `lena.jpg` (or create synthetic image if not found)
2. Add salt & pepper noise 
3. Apply median filtering
4. Display original, noisy, and filtered results

### C++ Build
For simd implementation (linux):
```bash
cd simd
make pixi-build
make pixi-run
```

for cuda implementation (linux):
```bash
cd cuda_implementation
make fix-cuda
make pixi-build
make pixi-run
```