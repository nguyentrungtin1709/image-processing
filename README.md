# Label Detection using Image Processing

A Python project for detecting labels on t-shirt images using multi-channel, multi-threshold image processing techniques with OpenCV.

## Overview

This project provides an automated solution for detecting rectangular label regions on product images. The detection algorithm processes images through multiple color channels (B, G, R) and applies various threshold values to find consistent rectangular patterns that likely represent labels.

### Key Features

- Multi-channel analysis (Blue, Green, Red)
- Multi-threshold binary processing
- Automatic rectangle detection and grouping
- Batch processing with CSV report generation
- Configurable detection parameters

## Getting Started

### Prerequisites

- Git
- Python 3.12+

### Installation

1. Clone the repository:

```bash
git clone https://github.com/nguyentrungtin1709/image-processing.git
```

2. Navigate to the project directory:

```bash
cd image-processing
```

3. Create a virtual environment:

```bash
python -m venv .venv
```

4. Activate the virtual environment:

**Windows:**

```powershell
.venv\Scripts\activate
```

**Linux/MacOS:**

```bash
source .venv/bin/activate
```

5. Install required packages:

```bash
pip install -r requirements.txt
```

## Usage

### Running Label Detection

Place your images in the `input/` folder, then run the detection script:

**Windows:**

```powershell
python scripts\run_label_detection.py
```

**Linux/MacOS:**

```bash
python scripts/run_label_detection.py
```

### Output

After processing, results will be saved to the `output/` folder:

- Processed images with bounding boxes drawn around detected labels
- `output/report/details.csv` - Detection results for each image (filename, number of labels, processing time)
- `output/report/summary.csv` - Summary statistics (total images, average time, etc.)

### Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff, .tif)

## Project Structure

```
image-processing/
├── core/
│   └── label_detector.py      # Core detection algorithm
├── scripts/
│   ├── run_label_detection.py # Batch processing script
│   ├── convert_structure.py   # Utility for folder conversion
│   └── jpeg_quality_simulator.py # JPEG compression simulator
├── input/                     # Input images folder
├── output/                    # Output results folder
└── requirements.txt           # Python dependencies
```

## License

This project is provided as-is for educational and research purposes.
