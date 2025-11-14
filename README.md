# Safety Kit Detection System

A real-time computer vision system that detects Personal Protective Equipment (PPE) on workers, including:
- **Helmet** (Hard Hat)
- **Safety Vest**
- **Gloves**
- **Safety Glasses**

## Features

- ✅ Real-time detection using laptop/webcam
- ✅ Advanced false positive prevention (distinguishes between actual PPE and similar items like hair, shirts, beards)
- ✅ Image enhancement for clearer camera feed
- ✅ Temporal consistency checking to avoid false detections
- ✅ Visual status panel showing detected/missing equipment
- ✅ High accuracy detection with strict validation

## Requirements

- Python 3.8 or higher
- Webcam/Laptop camera
- Windows/Linux/MacOS

## Installation

1. **Clone or download this project**

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Download YOLOv8 weights (automatic on first run)**
   - The system will automatically download YOLOv8n weights on first run
   - For better accuracy, consider training a custom model (see below)

## Usage

### Basic Usage (General YOLO Model)

Run the improved detection system:
```bash
python improved_detector.py
```

Or use the standard version:
```bash
python safety_detection.py
```

### Using a Custom Trained Model

If you have a custom-trained PPE detection model:
```bash
python improved_detector.py path/to/your/model.pt
```

## Training a Custom Model (Recommended for Best Accuracy)

For accurate detection without false positives, you should train a custom YOLOv8 model:

### Step 1: Prepare Dataset

1. Collect images of workers with and without PPE
2. Label images using tools like:
   - [LabelImg](https://github.com/tzutalin/labelImg)
   - [Roboflow](https://roboflow.com/)
   - [CVAT](https://github.com/openvinotoolkit/cvat)

3. Organize dataset in YOLO format:
```
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── dataset.yaml
```

### Step 2: Create dataset.yaml

```yaml
path: ./dataset
train: images/train
val: images/val
test: images/test

nc: 5
names:
  0: helmet
  1: vest
  2: gloves
  3: safety_glasses
  4: person
```

### Step 3: Train the Model

```bash
python train_ppe_model.py
```

Or modify `train_ppe_model.py` to specify your dataset path and training parameters.

### Step 4: Use Trained Model

After training, use your custom model:
```bash
python improved_detector.py runs/detect/ppe_detector/weights/best.pt
```

## How It Works

### False Positive Prevention

The system uses multiple techniques to prevent false detections:

1. **Visual Analysis:**
   - **Helmet vs Hair**: Analyzes color uniformity, edge density, and shape
   - **Vest vs Shirt**: Checks for yellow/orange colors and brightness typical of safety vests
   - **Glasses vs Beard/Face**: Detects frame structure, symmetry, and lens patterns
   - **Gloves**: Texture and position analysis

2. **Temporal Consistency:**
   - Tracks detections over multiple frames
   - Only confirms detections that appear consistently
   - Filters out sporadic false positives

3. **Confidence Thresholding:**
   - Uses strict confidence thresholds (default: 0.7)
   - Requires high confidence for single-frame detections
   - Lower threshold for consistent multi-frame detections

### Image Enhancement

The system applies multiple enhancement techniques:
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Bilateral filtering for noise reduction
- Unsharp masking for sharpness
- Gamma correction for better visibility

## Controls

- **'q'**: Quit the application
- **'s'**: Save current frame as image

## Troubleshooting

### Camera Not Opening
- Ensure no other application is using the camera
- Try changing camera index: Modify `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)`

### Poor Detection Accuracy
1. **Train a custom model** - The general YOLO model may not detect PPE items accurately
2. **Improve lighting** - Ensure good lighting conditions
3. **Adjust confidence threshold** - Modify `confidence_threshold` in the code
4. **Collect more training data** - More diverse training data improves accuracy

### False Positives
- The system includes advanced filtering, but for best results:
  - Train on your specific environment
  - Adjust validation thresholds in `FalsePositiveFilter` class
  - Increase confidence threshold

### Performance Issues
- Reduce frame processing rate (modify `frame_count % 2` to `frame_count % 3` or higher)
- Use smaller YOLO model (YOLOv8n instead of YOLOv8s/m/l/x)
- Reduce camera resolution

## Model Training Tips

1. **Dataset Size**: Aim for at least 500-1000 images per class
2. **Diversity**: Include various:
   - Lighting conditions
   - Angles and poses
   - Backgrounds
   - Equipment types and colors
3. **Negative Samples**: Include images without PPE to reduce false positives
4. **Augmentation**: Use data augmentation (rotation, brightness, etc.)
5. **Validation**: Keep 20% of data for validation

## Project Structure

```
CEP-Project/
├── improved_detector.py      # Main improved detection system
├── safety_detection.py        # Standard detection system
├── train_ppe_model.py         # Training script
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Future Improvements

- [ ] Support for multiple workers detection
- [ ] Alert system for missing PPE
- [ ] Database logging of violations
- [ ] Mobile app integration
- [ ] Edge device deployment (Raspberry Pi, Jetson Nano)

## License

This project is provided as-is for educational and commercial use.

## Support

For issues or questions:
1. Check the troubleshooting section
2. Ensure you have a custom-trained model for best results
3. Review the code comments for configuration options

## Notes

⚠️ **Important**: The general YOLO model may not detect PPE items accurately. For production use, **train a custom model** with your specific dataset for best results.

The system is designed to be highly accurate and prevent false positives, but accuracy depends on:
- Quality of training data
- Lighting conditions
- Camera quality
- Model training parameters

