# AIOT Wafer Detect

A simple command-line tool for detecting defects on semiconductor wafer images using image processing techniques with Python, OpenCV, and NumPy.

## Features

- Simple command-line interface for wafer defect detection
- Configurable detection parameters (threshold, minimum defect area)
- Automatic defect counting and localization
- Visual output with annotated defects
- Detailed defect information (area, center coordinates)

## Requirements

- Python 3.6+
- OpenCV (opencv-python)
- NumPy

## Installation

1. Clone the repository:
```bash
git clone https://github.com/E-Zicoo/AIOT_Wafer_Detect.git
cd AIOT_Wafer_Detect
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python wafer_detect.py input_image.jpg
```

### Save Visualization

```bash
python wafer_detect.py input_image.jpg -o output_image.jpg
```

### Advanced Usage with Custom Parameters

```bash
python wafer_detect.py input_image.jpg -t 100 -m 30 -o result.jpg -v
```

### Command-Line Options

- `input`: Input wafer image path (required)
- `-o, --output`: Output image path for visualization (optional)
- `-t, --threshold`: Binary threshold value, 0-255 (default: 127)
- `-m, --min-area`: Minimum defect area in pixels (default: 50)
- `-v, --verbose`: Enable verbose output

### Examples

Detect defects with default parameters:
```bash
python wafer_detect.py wafer.jpg
```

Detect with custom threshold and save result:
```bash
python wafer_detect.py wafer.jpg -t 150 -o detected.jpg
```

Detect small defects with verbose output:
```bash
python wafer_detect.py wafer.jpg -m 20 -v
```

## How It Works

The tool processes wafer images through the following steps:

1. **Image Loading**: Loads the input wafer image
2. **Preprocessing**: Converts to grayscale and applies Gaussian blur to reduce noise
3. **Thresholding**: Applies binary threshold to separate defects from background
4. **Contour Detection**: Finds contours representing potential defects
5. **Filtering**: Filters defects based on minimum area
6. **Analysis**: Calculates defect properties (area, center, bounding box)
7. **Visualization**: Optionally generates annotated output image

## Output Format

The tool provides:
- Total count of detected defects
- For each defect:
  - Center coordinates (x, y)
  - Area in pixels
  - Bounding box (when using visualization)

Example output:
```
==================================================
Wafer Defect Detection Results
==================================================
Total defects found: 3

Defect Details:
  Defect 1: Center=(245, 180), Area=156 pixels
  Defect 2: Center=(420, 310), Area=98 pixels
  Defect 3: Center=(580, 450), Area=203 pixels
==================================================
```

## Testing

To test the tool, you can create a simple test image or use your own wafer images:

```bash
# Test with help message
python wafer_detect.py --help

# Test with a sample image (you'll need to provide your own)
python wafer_detect.py sample_wafer.jpg -o result.jpg -v
```

## Technologies Used

- **Python**: Core programming language
- **OpenCV**: Image processing and computer vision
- **NumPy**: Numerical operations

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
