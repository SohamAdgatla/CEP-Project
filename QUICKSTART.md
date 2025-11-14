# Quick Start Guide

## Installation (5 minutes)

1. **Install Python dependencies:**
```bash
python setup.py
```

Or manually:
```bash
pip install -r requirements.txt
```

2. **Run the detection system:**
```bash
python improved_detector.py
```

## Important Note

⚠️ **The system uses a general YOLO model by default**, which may not detect PPE items accurately. 

For **accurate detection**, you need to:

### Option 1: Use Pre-trained PPE Model (if available)
Download a pre-trained PPE detection model and run:
```bash
python improved_detector.py path/to/ppe_model.pt
```

### Option 2: Train Your Own Model (Recommended)

1. **Collect Dataset:**
   - Gather images of workers with/without PPE
   - Use tools like [LabelImg](https://github.com/tzutalin/labelImg) to label images
   - Label classes: helmet, vest, gloves, safety_glasses, person

2. **Organize Dataset:**
```
dataset/
├── images/
│   ├── train/  (80% of images)
│   ├── val/    (10% of images)
│   └── test/   (10% of images)
└── dataset.yaml
```

3. **Create dataset.yaml:**
```yaml
path: ./dataset
train: images/train
val: images/val

nc: 5
names:
  0: helmet
  1: vest
  2: gloves
  3: safety_glasses
  4: person
```

4. **Train Model:**
```bash
python train_ppe_model.py
```

5. **Use Trained Model:**
```bash
python improved_detector.py runs/detect/ppe_detector/weights/best.pt
```

## Controls

- **'q'**: Quit
- **'s'**: Save current frame

## Troubleshooting

**Camera not working?**
- Close other apps using the camera
- Try: `cv2.VideoCapture(1)` instead of `0` in the code

**No detections?**
- This is expected with the general YOLO model
- Train a custom model for accurate detection

**False positives?**
- The system includes advanced filtering
- Adjust `confidence_threshold` in code (default: 0.7)
- Train on your specific environment

## Next Steps

1. Read `README.md` for detailed documentation
2. Train a custom model for your environment
3. Adjust detection parameters as needed
4. Integrate with your safety monitoring system

