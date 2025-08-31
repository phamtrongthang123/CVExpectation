# Makefile for SIMD Median Filter

# Compiler
CXX = g++

# Target executable
TARGET = median_filter_simd

# Source files
SOURCES = simd.cpp

# Object files
OBJECTS = $(SOURCES:.cpp=.o)

# Compiler flags
CXXFLAGS = -std=c++17 -O3 -march=native -mavx -mavx2 -mfma -fopenmp -Wall -Wextra

# OpenCV flags - try pkg-config first, fallback to conda environment
OPENCV_CFLAGS = $(shell pkg-config --cflags opencv4 2>/dev/null || pkg-config --cflags opencv 2>/dev/null || echo "-I$$CONDA_PREFIX/include")
OPENCV_LIBS = $(shell pkg-config --libs opencv4 2>/dev/null || pkg-config --libs opencv 2>/dev/null || echo "-L$$CONDA_PREFIX/lib -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui")

# Include directories
INCLUDES = $(OPENCV_CFLAGS)

# Libraries
LIBS = $(OPENCV_LIBS) -fopenmp

# Default target
all: $(TARGET)

# Build target
$(TARGET): $(OBJECTS)
	$(CXX) $(OBJECTS) -o $(TARGET) $(LIBS)

# Compile source files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Clean build artifacts
clean:
	rm -f $(OBJECTS) $(TARGET) *.png

# Install dependencies using pixi
install-deps:
	cd .. && pixi install

# Run the program
run: $(TARGET)
	./$(TARGET)

# Build in pixi environment
pixi-build:
	cd .. && pixi run bash -c "cd cpp && make $(TARGET)"

# Run the program in pixi environment
pixi-run: pixi-build
	cd .. && pixi run bash -c "cd cpp && ./$(TARGET)"

# Debug build
debug: CXXFLAGS = -std=c++17 -g -O0 -march=native -mavx -mavx2 -mfma -fopenmp -Wall -Wextra -DDEBUG
debug: $(TARGET)

# Profile build
profile: CXXFLAGS = -std=c++17 -O3 -march=native -mavx -mavx2 -mfma -fopenmp -Wall -Wextra -pg
profile: LIBS += -pg
profile: $(TARGET)

# Check if OpenCV is installed
check-opencv:
	@echo "Checking OpenCV installation..."
	@pkg-config --exists opencv4 && echo "OpenCV 4.x found via pkg-config" || echo "OpenCV 4.x not found via pkg-config"
	@pkg-config --exists opencv && echo "OpenCV 3.x found via pkg-config" || echo "OpenCV 3.x not found via pkg-config"
	@echo "Checking conda/pixi environment:"
	@cd .. && pixi run bash -c "echo 'CONDA_PREFIX: '\$$CONDA_PREFIX && ls \$$CONDA_PREFIX/lib | grep -i opencv || echo 'No OpenCV libs found in conda env'"

# Show compiler and linker flags
show-flags:
	@echo "Compiler flags: $(CXXFLAGS) $(INCLUDES)"
	@echo "Linker flags: $(LIBS)"

# Help target
help:
	@echo "Available targets:"
	@echo "  all          - Build the executable (default)"
	@echo "  clean        - Remove build artifacts"
	@echo "  run          - Build and run the program"
	@echo "  pixi-build   - Build the program in pixi environment"
	@echo "  pixi-run     - Build and run the program in pixi environment"
	@echo "  debug        - Build with debug symbols"
	@echo "  profile      - Build with profiling support"
	@echo "  install-deps - Install required dependencies using pixi"
	@echo "  check-opencv - Check if OpenCV is properly installed"
	@echo "  show-flags   - Display compiler and linker flags"
	@echo "  help         - Show this help message"

.PHONY: all clean run pixi-build pixi-run debug profile install-deps check-opencv show-flags help
