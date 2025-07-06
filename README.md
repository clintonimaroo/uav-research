# UAV Disaster Detection System

A lightweight PyTorch-based disaster detection system designed for autonomous UAV deployment. This system enables drones to detect disasters mid-flight and trigger autonomous rerouting for emergency response.

## 🎯 Project Overview

This project implements a computer vision pipeline for real-time disaster detection from aerial imagery, specifically designed for UAV integration. The system can detect various disaster types including fires, collapsed buildings, floods, and traffic incidents from drone camera feeds.

### Key Features

- **Lightweight Models**: MobileNetV2, EfficientNet-B0, and custom CNN architectures optimized for UAV deployment
- **Real-time Inference**: Optimized for real-time processing on drone hardware
- **Multiple Export Formats**: TorchScript and ONNX support for cross-platform deployment
- **Comprehensive Evaluation**: Detailed metrics, confusion matrices, and performance analysis
- **UAV Integration Ready**: Demo scripts showing integration with UAV control systems

## 📊 Dataset

The system is trained on the **AIDER (Aerial Image Database for Emergency Response)** dataset, which contains:

- **Fire**: 523 images
- **Collapsed Buildings**: 513 images  
- **Flooded Areas**: 528 images
- **Traffic Incidents**: 487 images
- **Normal**: 4,392 images

## 🏗️ Architecture

### Supported Models

1. **MobileNetV2**: Optimized for mobile/edge deployment (13.4M parameters)
2. **EfficientNet-B0**: Better accuracy with reasonable size (5.3M parameters)
3. **Lightweight CNN**: Custom architecture for ultra-fast inference (0.5M parameters)

### Model Comparison

| Model | Parameters | Size (MB) | Suitable for UAV |
|-------|------------|-----------|------------------|
| MobileNetV2 | 3.5M | 13.4 | ✓ |
| EfficientNet-B0 | 5.3M | 20.1 | ✓ |
| Lightweight CNN | 0.5M | 2.1 | ✓ |

## 🚀 Quick Start

### Installation

```bash
git clone <your-repo-url>
cd uav-research
pip install -r requirements.txt
```

### Dataset Analysis

```bash
python dataset_analysis.py
```

### Training

```bash
# Fire detection (binary classification)
python train.py --model mobilenet_v2 --classes fire normal --epochs 30

# Multi-class disaster detection
python train.py --model efficientnet_b0 --classes fire collapsed_building flooded_areas normal --epochs 50

# Using configuration files
python train.py --config configs/fire_detection.yaml
```

### Evaluation

```bash
python evaluate.py --model_path checkpoints/best_model.pth --save_dir evaluation_results
```

### Model Export

```bash
# Export all formats
python export_model.py --model_path checkpoints/best_model.pth --export_dir exported_models

# Export specific format
python export_model.py --model_path checkpoints/best_model.pth --format onnx
```

### Demo

```bash
# Webcam demo
python demo.py --model_path checkpoints/best_model.pth --mode webcam

# Video file processing
python demo.py --model_path checkpoints/best_model.pth --mode video --input_path video.mp4

# Batch image processing
python demo.py --model_path checkpoints/best_model.pth --mode batch --input_path image_folder/

# UAV integration example
python demo.py --model_path checkpoints/best_model.pth --mode uav_example
```

## 📁 Project Structure

```
uav-research/
├── dataset/                    # AIDER dataset
│   └── aider_dataset/
├── configs/                    # Training configurations
├── checkpoints/               # Model checkpoints
├── exported_models/           # Exported models for deployment
├── evaluation_results/        # Evaluation outputs
├── dataset_analysis.py        # Dataset analysis script
├── disaster_dataset.py        # PyTorch dataset implementation
├── models.py                  # Model architectures
├── config.py                  # Configuration management
├── train.py                   # Training pipeline
├── evaluate.py               # Evaluation script
├── export_model.py           # Model export utilities
├── demo.py                   # Demo and UAV integration
├── utils.py                  # Utility functions
├── requirements.txt          # Dependencies
└── README.md                 # This file
```

## 🔧 Configuration

The system uses YAML configuration files for easy experiment management:

```yaml
# Example: configs/fire_detection.yaml
dataset_path: "dataset/aider_ dataset/"
classes: ["fire", "normal"]
model_name: "mobilenet_v2"
batch_size: 32
num_epochs: 30
learning_rate: 0.001
freeze_backbone: true
augment: true
```

### Pre-configured Experiments

- `fire_detection.yaml`: Binary fire detection
- `multi_disaster.yaml`: Multi-class disaster detection
- `lightweight.yaml`: Ultra-fast model for resource-constrained UAVs
- `all_classes.yaml`: Full disaster classification

## 📈 Training Pipeline

The training pipeline includes:

- **Data Augmentation**: Random crops, flips, rotation, color jittering
- **Class Balancing**: Weighted loss functions for imbalanced datasets
- **Early Stopping**: Prevents overfitting with patience-based stopping
- **Learning Rate Scheduling**: StepLR scheduler for optimal convergence
- **Tensorboard Logging**: Real-time training visualization
- **Model Checkpointing**: Automatic saving of best models

## 🎯 UAV Integration

### Real-time Disaster Detection

```python
from demo import DisasterDetectionDemo

# Initialize detector
detector = DisasterDetectionDemo('best_model.pth')

# Process UAV camera feed
result = detector.predict_single_image(drone_frame)

if result['should_alert']:
    # Trigger emergency protocols
    uav.reroute_to_safe_area()
    uav.alert_ground_control(result)
```

### Integration Features

- **Confidence Thresholding**: Configurable thresholds for disaster alerts
- **Real-time Processing**: Optimized for 10+ FPS on drone hardware
- **Emergency Protocols**: Automatic rerouting and alert systems
- **Telemetry Integration**: GPS coordinates and flight data logging

## 📊 Performance Metrics

### Evaluation Metrics

- **Accuracy**: Overall classification accuracy
- **Precision/Recall**: Per-class performance metrics
- **F1-Score**: Balanced performance measure
- **ROC AUC**: Area under the ROC curve
- **Inference Time**: Processing speed for real-time deployment

### Expected Performance

| Model | Accuracy | Inference Time | FPS |
|-------|----------|----------------|-----|
| MobileNetV2 | 85-90% | 15-20ms | 50-65 |
| EfficientNet-B0 | 88-93% | 25-30ms | 33-40 |
| Lightweight CNN | 80-85% | 8-12ms | 80-125 |

## 🛠️ Advanced Features

### Model Export Formats

- **TorchScript**: For PyTorch-based deployment
- **ONNX**: For cross-platform inference
- **Model Metadata**: Class mappings and preprocessing info

### Deployment Optimization

- **Quantization Ready**: Models prepared for INT8 quantization
- **TensorRT Compatible**: ONNX models work with TensorRT
- **Edge Deployment**: Optimized for NVIDIA Jetson and similar platforms

## 🔬 Research Applications

This system supports research in:

- **Autonomous UAV Navigation**: Emergency response and path planning
- **Computer Vision**: Aerial image classification and object detection
- **Edge AI**: Lightweight model deployment on resource-constrained devices
- **Disaster Response**: Real-time disaster monitoring and assessment

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Contact

For questions about UAV integration or research collaboration, please contact:
- Email: [your-email@university.edu]
- Lab: [Your Research Lab]
- University: [Your University]

## 🙏 Acknowledgments

- AIDER Dataset contributors
- PyTorch and torchvision teams
- UAV research community
- [Your Professor/Advisor Name]

---

**Note**: This system is designed for research purposes. For production UAV deployment, additional safety measures, regulatory compliance, and extensive testing are required. 