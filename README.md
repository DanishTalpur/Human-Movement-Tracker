# Human Movement Tracker

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![YOLO](https://img.shields.io/badge/YOLO-v8-red.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.6+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**Advanced AI-powered human tracking with real-time movement direction analysis**

[Features](#features) • [Quick Start](#quick-start) • [Demo](#demo) • [Documentation](#documentation)

</div>

---

## Project Description

An intelligent computer vision system that leverages state-of-the-art YOLOv8 object detection to track multiple people in video streams while providing real-time movement direction analysis. The system features smoothed trajectory tracking, persistent directional arrows, and robust multi-person tracking capabilities perfect for surveillance, crowd analysis, and behavioral studies.

## Features

- **YOLOv8-Powered Detection**: Cutting-edge object detection for accurate person identification
- **Real-time Tracking**: Persistent tracking across frames with unique ID assignment
- **Direction Analysis**: Smoothed movement vectors with visual directional indicators
- **Visual Overlays**: Customizable bounding boxes and persistent arrow overlays
- **Video Processing**: Support for various video formats with configurable output
- **Easy Training**: Jupyter notebooks for custom model training on your datasets
- **High Performance**: Optimized for real-time processing with GPU acceleration

## Quick Start

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/human-movement-tracker.git
   cd human-movement-tracker
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # source .venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the tracker**
   ```bash
   python main.py
   ```

## Demo

The system processes input videos and generates annotated outputs showing:
- **Purple bounding boxes** around detected people
- **Pink directional arrows** indicating movement direction
- **Smooth trajectory tracking** with persistence

### Sample Output
```
Input:  background-video-people-walking_1080p.mp4
Output: videos/output_tracking_people.avi
```

## Project Structure

```
human-movement-tracker/
├── videos/                    # Sample videos and outputs
│   ├── background-video-people-walking_1080p.mp4
│   └── output_tracking_people.avi
├── Training/                  # Model training resources
│   ├── People-Detection-10/  # Dataset
│   ├── runs/                 # Training outputs
│   ├── training.ipynb        # Training notebook
│   └── testing.ipynb         # Testing notebook
├── main.py                   # Main tracking script
├── requirements.txt          # Dependencies
└── README.md                # This file
```

## Configuration

### Environment Variables
Create a `.env` file for Roboflow integration:
```env
ROBOFLOW_API_KEY=your_api_key_here
```

### Customization Options
- **Arrow persistence**: Modify `arrow_persistence` (default: 10 frames)
- **Movement threshold**: Adjust `min_speed` (default: 2 pixels)
- **Visual styling**: Customize colors and thickness in `main.py`

## Training Your Own Model

1. **Open training notebook**
   ```bash
   jupyter notebook Training/training.ipynb
   ```

2. **Follow the step-by-step guide** to:
   - Download datasets from Roboflow
   - Configure training parameters
   - Train custom YOLOv8 models
   - Evaluate model performance

## Performance

- **Detection Speed**: ~30-60 FPS (GPU dependent)
- **Accuracy**: >95% person detection rate
- **Memory Usage**: ~2-4GB VRAM
- **Supported Formats**: MP4, AVI, MOV, and more

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Ultralytics](https://ultralytics.com/) for YOLOv8
- [Roboflow](https://roboflow.com/) for dataset management
- [OpenCV](https://opencv.org/) for computer vision utilities

## Support

- **Bug Reports**: [GitHub Issues](https://github.com/yourusername/human-movement-tracker/issues)
- **Feature Requests**: [GitHub Discussions](https://github.com/yourusername/human-movement-tracker/discussions)
- **Contact**: [danishshuja11@gmail.com](mailto:danishshuja11@gmail.com)

---

<div align="center">

**Star this repository if you find it helpful!**

Made with ❤️ for the computer vision community

</div>
