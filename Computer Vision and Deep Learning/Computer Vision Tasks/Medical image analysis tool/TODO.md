# Medical Image Analysis Tool - TODO

## Project Overview
Develop tools for analyzing medical images (X-rays, MRI, CT scans, etc.).

## Completed Tasks
- [x] **CT Processing Tutorials Created** - Added comprehensive tutorials for CT image reading and plotting
  - `ct_tutorial_simpleitk.py` - Basic CT processing with SimpleITK
  - `ct_tutorial_monai.py` - Advanced CT processing with MONAI
  - `requirements.txt` - All necessary dependencies
- [x] Set up project structure
- [x] Implement basic DICOM/medical file handling (via tutorials)
- [x] Create visualization tools (CT-specific)
- [x] Document medical use cases (CT processing)

## In Progress Tasks
- [ ] Research medical imaging datasets
- [ ] Design comprehensive preprocessing pipeline
- [ ] Implement segmentation/classification models
- [ ] Add medical-specific evaluation metrics
- [ ] Ensure HIPAA compliance considerations

## Future Enhancements
- [ ] Add support for other modalities (MRI, X-ray, ultrasound)
- [ ] Implement advanced segmentation algorithms
- [ ] Add automated pathology detection
- [ ] Create interactive web interface for image viewing
- [ ] Add DICOM metadata extraction and analysis
- [ ] Implement image registration and fusion capabilities
- [ ] Add support for 4D imaging (time series)

## Tutorial Features Implemented

### SimpleITK Tutorial (`ct_tutorial_simpleitk.py`)
- CT image loading and metadata inspection
- Orthogonal view visualization (axial, sagittal, coronal)
- Intensity histogram analysis with HU reference values
- Windowing techniques for different tissue types
- Image resampling and preprocessing
- Synthetic data generation for demonstration

### MONAI Tutorial (`ct_tutorial_monai.py`)
- Advanced transform pipelines for CT preprocessing
- Integration with PyTorch for deep learning workflows
- Data augmentation techniques for medical imaging
- Dataset and DataLoader creation
- 3D visualization capabilities
- Intensity normalization and standardization

## Notes
- Focus on specific medical condition (pneumonia, tumors, etc.)
- Use pydicom for DICOM file handling and SimpleITK for medical image processing
- Consider class imbalance with scikit-learn's imbalanced-learn library
- PyTorch Lightning for medical model training and TorchMetrics for medical-specific metrics
- **New**: Both tutorials include synthetic data generation for immediate testing
- **New**: MONAI integration provides state-of-the-art medical AI capabilities
