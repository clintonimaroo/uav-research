import os
import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import time
from typing import Dict, List, Tuple, Optional

from config import TrainingConfig
from models import create_model
from disaster_dataset import get_transforms
from utils import load_checkpoint

class DisasterDetectionDemo:
    """Demo class for disaster detection with UAV integration focus"""
    
    def __init__(self, model_path: str, config_path: str = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        if config_path:
            self.config = TrainingConfig.load_config(config_path)
        else:
            self.config = self.checkpoint.get('config', None)
            if self.config is None:
                raise ValueError("No configuration found. Please provide config_path.")
        
        self.class_to_idx = self.checkpoint.get('class_to_idx', {})
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
        self.model = create_model(
            model_name=self.config.model_name,
            num_classes=self.config.num_classes
        )
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        _, self.transform = get_transforms(self.config.input_size, augment=False)
        
        self.disaster_threshold = 0.7  # Confidence threshold for disaster detection
        self.alert_classes = [cls for cls in self.class_to_idx.keys() if cls != 'normal']
        
        print(f"✓ Loaded model: {self.config.model_name}")
        print(f"✓ Classes: {list(self.class_to_idx.keys())}")
        print(f"✓ Device: {self.device}")
        print(f"✓ Disaster threshold: {self.disaster_threshold}")
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for inference"""
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        pil_image = Image.fromarray(image)
        
        tensor = self.transform(pil_image).unsqueeze(0)
        return tensor.to(self.device)
    
    def predict_single_image(self, image: np.ndarray) -> Dict:
        """Predict disaster class for a single image"""
        input_tensor = self.preprocess_image(image)
        
        with torch.no_grad():
            start_time = time.time()
            outputs = self.model(input_tensor)
            inference_time = time.time() - start_time
            
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            predicted_class_idx = np.argmax(probabilities)
            predicted_class = self.idx_to_class[predicted_class_idx]
            confidence = probabilities[predicted_class_idx]
        
        result = {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': {self.idx_to_class[i]: prob for i, prob in enumerate(probabilities)},
            'inference_time_ms': inference_time * 1000,
            'is_disaster': predicted_class in self.alert_classes,
            'should_alert': predicted_class in self.alert_classes and confidence > self.disaster_threshold
        }
        
        return result
    
    def process_video_stream(self, video_path: str = None, save_output: bool = False):
        """Process video stream (webcam or file) for real-time disaster detection"""
        if video_path is None:
            cap = cv2.VideoCapture(0) 
            print("Using webcam...")
        else:
            cap = cv2.VideoCapture(video_path)
            print(f"Processing video: {video_path}")
        
        if not cap.isOpened():
            raise ValueError("Could not open video source")
        
        if save_output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter('demo_output.mp4', fourcc, fps, (width, height))
        
        frame_count = 0
        total_time = 0
        
        print("Starting video processing... Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            if frame_count % 3 == 0:  # Process every 3rd frame
                result = self.predict_single_image(frame)
                total_time += result['inference_time_ms']
                
                frame = self.draw_prediction_on_frame(frame, result)
                
                if result['should_alert']:
                    print(f"🚨 DISASTER DETECTED: {result['predicted_class']} "
                          f"(confidence: {result['confidence']:.2f})")
                    self.trigger_uav_action(result)
            
            cv2.imshow('Disaster Detection Demo', frame)
            
            if save_output:
                out.write(frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        if save_output:
            out.release()
        cv2.destroyAllWindows()
        
        if frame_count > 0:
            avg_inference_time = total_time / (frame_count // 3)
            print(f"\nProcessing complete:")
            print(f"  Frames processed: {frame_count}")
            print(f"  Average inference time: {avg_inference_time:.2f} ms")
            print(f"  Average FPS: {1000 / avg_inference_time:.1f}")
    
    def draw_prediction_on_frame(self, frame: np.ndarray, result: Dict) -> np.ndarray:
        """Draw prediction results on frame"""
        height, width = frame.shape[:2]
        
        if result['should_alert']:
            color = (0, 0, 255)  
            status = "DISASTER DETECTED"
        elif result['is_disaster']:
            color = (0, 165, 255)  
            status = "POSSIBLE DISASTER"
        else:
            color = (0, 255, 0) 
            status = "NORMAL"
        
        cv2.rectangle(frame, (10, 10), (width - 10, 120), color, 2)
        cv2.rectangle(frame, (10, 10), (width - 10, 120), color, -1)
        cv2.rectangle(frame, (10, 10), (width - 10, 120), (255, 255, 255), -1)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, status, (20, 35), font, 0.8, color, 2)
        cv2.putText(frame, f"Class: {result['predicted_class']}", (20, 60), font, 0.6, (0, 0, 0), 1)
        cv2.putText(frame, f"Confidence: {result['confidence']:.2f}", (20, 80), font, 0.6, (0, 0, 0), 1)
        cv2.putText(frame, f"Time: {result['inference_time_ms']:.1f}ms", (20, 100), font, 0.6, (0, 0, 0), 1)
        
        y_start = 140
        bar_height = 20
        max_bar_width = 200
        
        for i, (class_name, prob) in enumerate(result['probabilities'].items()):
            y = y_start + i * (bar_height + 5)
            bar_width = int(prob * max_bar_width)
            
            # Draw bar background
            cv2.rectangle(frame, (20, y), (20 + max_bar_width, y + bar_height), (200, 200, 200), -1)
            
            # Draw probability bar
            bar_color = (0, 0, 255) if class_name in self.alert_classes else (0, 255, 0)
            cv2.rectangle(frame, (20, y), (20 + bar_width, y + bar_height), bar_color, -1)
            
            # Draw text
            cv2.putText(frame, f"{class_name}: {prob:.2f}", (30, y + 15), font, 0.4, (0, 0, 0), 1)
        
        return frame
    
    def trigger_uav_action(self, result: Dict):
        """Simulate UAV action trigger (placeholder for actual UAV integration)"""
        action_data = {
            'timestamp': time.time(),
            'disaster_type': result['predicted_class'],
            'confidence': result['confidence'],
            'recommended_action': 'REROUTE_AND_INVESTIGATE',
            'priority': 'HIGH' if result['confidence'] > 0.8 else 'MEDIUM'
        }
        
        # In a real UAV system, this would:
        # 1. Send alert to ground control
        # 2. Trigger path planning algorithm
        # 3. Initiate emergency protocols
        # 4. Log incident for analysis
        
        print(f"UAV Action Triggered: {action_data}")
    
    def batch_process_images(self, image_dir: str, output_dir: str = None):
        """Process a batch of images and save results"""
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = [f for f in os.listdir(image_dir) 
                      if any(f.lower().endswith(ext) for ext in image_extensions)]
        
        results = []
        
        print(f"Processing {len(image_files)} images...")
        
        for i, image_file in enumerate(image_files):
            image_path = os.path.join(image_dir, image_file)
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Could not load image: {image_file}")
                continue
            
            # Predict
            result = self.predict_single_image(image)
            result['filename'] = image_file
            results.append(result)
            
            if output_dir:
                annotated_image = self.draw_prediction_on_frame(image.copy(), result)
                output_path = os.path.join(output_dir, f"annotated_{image_file}")
                cv2.imwrite(output_path, annotated_image)
            
            # Print progress
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(image_files)} images")
        
        # Save results summary
        self.save_batch_results(results, output_dir)
        
        return results
    
    def save_batch_results(self, results: List[Dict], output_dir: str = None):
        """Save batch processing results"""
        if not results:
            return
        
        # Calculate statistics
        total_images = len(results)
        disaster_detections = sum(1 for r in results if r['is_disaster'])
        high_confidence_alerts = sum(1 for r in results if r['should_alert'])
        avg_inference_time = np.mean([r['inference_time_ms'] for r in results])
        
        # Print summary
        print(f"\n=== Batch Processing Results ===")
        print(f"Total images processed: {total_images}")
        print(f"Disaster detections: {disaster_detections}")
        print(f"High confidence alerts: {high_confidence_alerts}")
        print(f"Average inference time: {avg_inference_time:.2f} ms")
        
        # Save detailed results
        if output_dir:
            import json
            results_file = os.path.join(output_dir, 'detection_results.json')
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Detailed results saved to: {results_file}")
    
    def create_uav_integration_example(self):
        """Show example of UAV integration code"""
        example_code = """
# UAV Integration Example
# This shows how to integrate the disaster detection model with UAV systems

import numpy as np
from your_uav_sdk import UAV, PathPlanner

class UAVDisasterDetector:
    def __init__(self, model_path, uav_instance):
        self.detector = DisasterDetectionDemo(model_path)
        self.uav = uav_instance
        self.path_planner = PathPlanner()
        
    def process_camera_feed(self):
        while self.uav.is_flying():
            # Get current camera frame
            frame = self.uav.get_camera_frame()
            
            # Run disaster detection
            result = self.detector.predict_single_image(frame)
            
            # Check for disaster
            if result['should_alert']:
                self.handle_disaster_detection(result)
            
            # Continue normal flight
            time.sleep(0.1)  # 10 FPS processing
    
    def handle_disaster_detection(self, result):
        # Get current position
        current_pos = self.uav.get_position()
        
        # Plan new route avoiding disaster area
        safe_route = self.path_planner.plan_safe_route(
            current_pos, 
            disaster_location=current_pos,
            disaster_type=result['predicted_class']
        )
        
        # Execute emergency procedures
        self.uav.send_alert_to_ground_control(result)
        self.uav.set_new_route(safe_route)
        self.uav.increase_altitude(50)  # Gain altitude for safety
        
        print(f"Emergency route activated due to {result['predicted_class']}")

# Usage
uav = UAV()
detector = UAVDisasterDetector('best_model.pth', uav)
detector.process_camera_feed()
"""
        
        print("=== UAV Integration Example ===")
        print(example_code)

def main():
    """Main demo function"""
    parser = argparse.ArgumentParser(description='Disaster Detection Demo')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--config_path', type=str, default=None,
                       help='Path to configuration file')
    parser.add_argument('--mode', type=str, choices=['webcam', 'video', 'batch', 'uav_example'],
                       default='webcam', help='Demo mode')
    parser.add_argument('--input_path', type=str, default=None,
                       help='Input video file or image directory')
    parser.add_argument('--output_dir', type=str, default='demo_output',
                       help='Output directory for results')
    parser.add_argument('--save_output', action='store_true',
                       help='Save processed video/images')
    
    args = parser.parse_args()
    
    # Create demo instance
    demo = DisasterDetectionDemo(args.model_path, args.config_path)
    
    # Run demo based on mode
    if args.mode == 'webcam':
        demo.process_video_stream(save_output=args.save_output)
    elif args.mode == 'video':
        if not args.input_path:
            raise ValueError("--input_path required for video mode")
        demo.process_video_stream(args.input_path, args.save_output)
    elif args.mode == 'batch':
        if not args.input_path:
            raise ValueError("--input_path required for batch mode")
        demo.batch_process_images(args.input_path, args.output_dir if args.save_output else None)
    elif args.mode == 'uav_example':
        demo.create_uav_integration_example()
    
    print("\nDemo completed!")

if __name__ == "__main__":
    main() 