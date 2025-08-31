# Makefile for CUDA Median Filter

# CUDA Compiler
NVCC = nvcc

# Target executable
TARGET = median_filter_cuda

# Source files
SOURCES = cuda.cu

# Object files
OBJECTS = $(SOURCES:.cu=.o)

# CUDA flags
NVCC_FLAGS = -std=c++17 -O3 -arch=sm_60 -arch=sm_70 -arch=sm_75 -arch=sm_80 --use_fast_math

# OpenCV flags - try pkg-config first, fallback to conda environment
OPENCV_CFLAGS = $(shell pkg-config --cflags opencv4 2>/dev/null || pkg-config --cflags opencv 2>/dev/null || echo "-I$$CONDA_PREFIX/include")
OPENCV_LIBS = $(shell pkg-config --libs opencv4 2>/dev/null || pkg-config --libs opencv 2>/dev/null || echo "-L$$CONDA_PREFIX/lib -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui")

# Include directories
INCLUDES = $(OPENCV_CFLAGS)

# Libraries
LIBS = $(OPENCV_LIBS) -lcudart

# Default target
all: $(TARGET)

# Build target
$(TARGET): $(OBJECTS)
	$(NVCC) $(OBJECTS) -o $(TARGET) $(LIBS)

# Compile CUDA source files
%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

# Clean build artifacts
clean:
	rm -f $(OBJECTS) $(TARGET) *.png

# Install dependencies using pixi
install-deps:
	cd .. && pixi install

# Fix CUDA version compatibility
fix-cuda:
	@echo "Fixing CUDA version compatibility..."
	cd .. && pixi remove cuda-toolkit cuda-nvcc
	cd .. && pixi add "cuda-toolkit>=11.0,<13" "cuda-nvcc>=11.0,<13"
	cd .. && pixi install

# Setup - copy required files
setup:
	cp ../lena.jpg .

# Build in pixi environment
pixi-build:
	cd .. && pixi run bash -c "cd cuda_implementation && make $(TARGET)"

# Run the program in pixi environment
pixi-run: setup pixi-build
	cd .. && pixi run bash -c "cd cuda_implementation && ./$(TARGET)"

# Debug build
debug: NVCC_FLAGS = -std=c++17 -g -G -O0 -arch=sm_60 -arch=sm_70 -arch=sm_75 -arch=sm_80
debug: $(TARGET)

# Profile build
profile: NVCC_FLAGS = -std=c++17 -O3 -arch=sm_60 -arch=sm_70 -arch=sm_75 -arch=sm_80 --use_fast_math -lineinfo
profile: $(TARGET)

# Check if CUDA is installed
check-cuda:
	@echo "Checking CUDA installation..."
	@echo "=== System CUDA Driver ==="
	@nvidia-smi 2>/dev/null | head -10 || echo "NVIDIA driver not found or not working"
	@echo ""
	@echo "=== CUDA Runtime Version ==="
	@cd .. && pixi run bash -c "nvcc --version 2>/dev/null || echo 'NVCC not found in pixi environment'"
	@echo ""
	@echo "=== Driver/Runtime Compatibility ==="
	@echo "CUDA 12.0+ requires driver >= 525.60.13"
	@echo "CUDA 11.8 requires driver >= 520.61.05" 
	@echo "CUDA 11.7 requires driver >= 515.48.07"
	@echo "CUDA 11.6 requires driver >= 510.47.03"
	@echo "CUDA 11.0 requires driver >= 450.80.02"

# Check if OpenCV is installed
check-opencv:
	@echo "Checking OpenCV installation..."
	@pkg-config --exists opencv4 && echo "OpenCV 4.x found via pkg-config" || echo "OpenCV 4.x not found via pkg-config"
	@pkg-config --exists opencv && echo "OpenCV 3.x found via pkg-config" || echo "OpenCV 3.x not found via pkg-config"
	@echo "Checking conda/pixi environment:"
	@cd .. && pixi run bash -c "echo 'CONDA_PREFIX: '\$$CONDA_PREFIX && ls \$$CONDA_PREFIX/lib | grep -i opencv || echo 'No OpenCV libs found in conda env'"

# Show compiler and linker flags
show-flags:
	@echo "NVCC flags: $(NVCC_FLAGS) $(INCLUDES)"
	@echo "Linker flags: $(LIBS)"

# Check GPU capabilities
check-gpu:
	@echo "Checking GPU capabilities..."
	@cd .. && pixi run bash -c "cd cuda_implementation && nvidia-smi -q -d computecap 2>/dev/null || echo 'Cannot detect GPU capabilities'"

# Help target
help:
	@echo "Available targets:"
	@echo "  all          - Build the executable (default)"
	@echo "  clean        - Remove build artifacts"
	@echo "  setup        - Copy required files (lena.jpg)"
	@echo "  pixi-build   - Build the program in pixi environment"
	@echo "  pixi-run     - Build and run the program in pixi environment"
	@echo "  debug        - Build with debug symbols"
	@echo "  profile      - Build with profiling support"
	@echo "  install-deps - Install required dependencies using pixi"
	@echo "  fix-cuda     - Fix CUDA version compatibility issues"
	@echo "  check-cuda   - Check if CUDA is properly installed"
	@echo "  check-opencv - Check if OpenCV is properly installed"
	@echo "  check-gpu    - Check GPU capabilities"
	@echo "  show-flags   - Display compiler and linker flags"
	@echo "  help         - Show this help message"

.PHONY: all clean setup pixi-build pixi-run debug profile install-deps fix-cuda check-cuda check-opencv check-gpu show-flags help
