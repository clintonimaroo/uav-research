import os
import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time
import json
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import threading
from collections import deque

import sys
sys.path.append('..')
from config import TrainingConfig
from models import create_model
from disaster_dataset import get_transforms
from utils import load_checkpoint

class CustomVideoDisasterAnalyzer:
    """Custom Video Analysis System for Real-time Disaster Detection"""
    
    def __init__(self, model_path: str, config_path: str = None):
        """Initialize the video analyzer with trained model"""
        print("🎬 Initializing Custom Video Disaster Analyzer...")
        
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
        
        self.confidence_threshold = 0.7
        self.disaster_classes = [cls for cls in self.class_to_idx.keys() if cls != 'normal']
        
        self.frame_buffer = deque(maxlen=30) 
        self.analysis_history = []
        
        self.process_every_n_frames = 2  
        
        print(f"✅ Model loaded: {self.config.model_name}")
        print(f"✅ Classes: {list(self.class_to_idx.keys())}")
        print(f"✅ Device: {self.device}")
        print(f"✅ Confidence threshold: {self.confidence_threshold}")
        
    def analyze_frame(self, frame: np.ndarray) -> Dict:
        """Analyze a single frame for disaster detection"""
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            rgb_frame = frame
            
        pil_image = Image.fromarray(rgb_frame)
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            start_time = time.time()
            outputs = self.model(input_tensor)
            inference_time = time.time() - start_time
            
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            predicted_idx = np.argmax(probabilities)
            predicted_class = self.idx_to_class[predicted_idx]
            confidence = probabilities[predicted_idx]
        
        result = {
            'timestamp': time.time(),
            'predicted_class': predicted_class,
            'confidence': float(confidence),
            'probabilities': {self.idx_to_class[i]: float(prob) for i, prob in enumerate(probabilities)},
            'inference_time_ms': inference_time * 1000,
            'is_disaster': predicted_class in self.disaster_classes,
            'alert_level': self._get_alert_level(predicted_class, confidence)
        }
        
        return result
    
    def _get_alert_level(self, predicted_class: str, confidence: float) -> str:
        """Determine alert level based on prediction and confidence"""
        if predicted_class == 'normal':
            return 'SAFE'
        elif confidence >= 0.9:
            return 'CRITICAL'
        elif confidence >= 0.7:
            return 'HIGH'
        elif confidence >= 0.5:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def create_analysis_overlay(self, frame: np.ndarray, result: Dict, frame_number: int) -> np.ndarray:
        """Create detailed analysis overlay on video frame"""
        overlay_frame = frame.copy()
        h, w = frame.shape[:2]
        
        color_map = {
            'CRITICAL': (0, 0, 255),    # Red
            'HIGH': (0, 165, 255),      # Orange
            'MEDIUM': (0, 255, 255),    # Yellow
            'LOW': (255, 255, 0),       # Cyan
            'SAFE': (0, 255, 0)         # Green
        }
        
        alert_color = color_map.get(result['alert_level'], (128, 128, 128))
        
        panel_height = 180
        cv2.rectangle(overlay_frame, (0, 0), (w, panel_height), (0, 0, 0), -1)
        cv2.rectangle(overlay_frame, (0, 0), (w, panel_height), alert_color, 3)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(overlay_frame, f"DISASTER ANALYSIS - FRAME {frame_number}", 
                   (20, 35), font, 0.8, (255, 255, 255), 2)
        
        status_text = f"STATUS: {result['alert_level']}"
        cv2.putText(overlay_frame, status_text, (20, 65), font, 0.7, alert_color, 2)
        
        class_text = f"DETECTED: {result['predicted_class'].upper()}"
        confidence_text = f"CONFIDENCE: {result['confidence']:.1%}"
        cv2.putText(overlay_frame, class_text, (20, 95), font, 0.6, (255, 255, 255), 2)
        cv2.putText(overlay_frame, confidence_text, (20, 120), font, 0.6, (255, 255, 255), 2)
        
        perf_text = f"INFERENCE: {result['inference_time_ms']:.1f}ms"
        cv2.putText(overlay_frame, perf_text, (20, 145), font, 0.5, (200, 200, 200), 1)
        
        bar_x = w - 300
        bar_y_start = 30
        bar_width = 250
        bar_height = 20
        
        cv2.putText(overlay_frame, "REAL-TIME PROBABILITIES:", 
                   (bar_x, bar_y_start), font, 0.5, (255, 255, 255), 1)
        
        for i, (class_name, prob) in enumerate(result['probabilities'].items()):
            y_pos = bar_y_start + 25 + (i * 25)
            
            cv2.rectangle(overlay_frame, (bar_x, y_pos), (bar_x + bar_width, y_pos + bar_height), 
                         (60, 60, 60), -1)
            
            prob_width = int(prob * bar_width)
            bar_color = (0, 0, 255) if class_name in self.disaster_classes else (0, 255, 0)
            cv2.rectangle(overlay_frame, (bar_x, y_pos), (bar_x + prob_width, y_pos + bar_height), 
                         bar_color, -1)
            
            cv2.putText(overlay_frame, f"{class_name}: {prob:.1%}", 
                       (bar_x + 5, y_pos + 15), font, 0.4, (255, 255, 255), 1)
        
        timeline_y = h - 40
        cv2.rectangle(overlay_frame, (0, timeline_y), (w, h), (0, 0, 0), -1)
        
        if self.frame_buffer:
            timeline_width = min(len(self.frame_buffer), 50)
            section_width = w // timeline_width
            
            for i, historical_result in enumerate(list(self.frame_buffer)[-timeline_width:]):
                x_start = i * section_width
                x_end = (i + 1) * section_width
                hist_color = color_map.get(historical_result.get('alert_level', 'SAFE'), (128, 128, 128))
                cv2.rectangle(overlay_frame, (x_start, timeline_y + 10), (x_end, h - 10), hist_color, -1)
        
        cv2.putText(overlay_frame, "ANALYSIS TIMELINE", (20, timeline_y + 25), font, 0.5, (255, 255, 255), 1)
        
        return overlay_frame
    
    def process_video(self, video_path: str, output_path: str = None, show_live: bool = True) -> Dict:
        """Process a video file with real-time disaster detection analysis"""
        print(f"🎬 Processing video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video Info: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"Output will be saved to: {output_path}")
        
        frame_count = 0
        processed_count = 0
        analysis_results = []
        start_time = time.time()
        
        print("Starting video analysis...")
        print("Press 'q' to quit, 'p' to pause, 'r' to resume")
        
        paused = False
        
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                if frame_count % self.process_every_n_frames == 0:
                    result = self.analyze_frame(frame)
                    result['frame_number'] = frame_count
                    
                    self.frame_buffer.append(result)
                    analysis_results.append(result)
                    processed_count += 1
                    
                    analyzed_frame = self.create_analysis_overlay(frame, result, frame_count)
                    
                    if processed_count % 30 == 0:  # Every 30 processed frames
                        progress = (frame_count / total_frames) * 100
                        print(f"Progress: {progress:.1f}% | Frame: {frame_count}/{total_frames} | "
                              f"Detected: {result['predicted_class']} ({result['confidence']:.1%})")
                else:
                    analyzed_frame = frame
                
                if show_live:
                    cv2.imshow('Custom Video Disaster Analysis', analyzed_frame)
                
                if output_path:
                    out.write(analyzed_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                paused = not paused
                print("⏸Paused" if paused else "▶Resumed")
        
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()
        
        processing_time = time.time() - start_time
        summary = self._generate_analysis_summary(analysis_results, processing_time)
        
        print("Video analysis complete!")
        self._print_analysis_summary(summary)
        
        return summary
    
    def _generate_analysis_summary(self, results: List[Dict], processing_time: float) -> Dict:
        """Generate comprehensive analysis summary"""
        if not results:
            return {}
        
        total_frames = len(results)
        disaster_frames = sum(1 for r in results if r['is_disaster'])
        
        class_counts = {}
        for result in results:
            cls = result['predicted_class']
            class_counts[cls] = class_counts.get(cls, 0) + 1
        
        alert_counts = {}
        for result in results:
            alert = result['alert_level']
            alert_counts[alert] = alert_counts.get(alert, 0) + 1
        
        confidences = [r['confidence'] for r in results]
        avg_confidence = np.mean(confidences)
        max_confidence = np.max(confidences)
        min_confidence = np.min(confidences)
        
        inference_times = [r['inference_time_ms'] for r in results]
        avg_inference_time = np.mean(inference_times)
        
        summary = {
            'total_frames_analyzed': total_frames,
            'disaster_frames_detected': disaster_frames,
            'disaster_percentage': (disaster_frames / total_frames) * 100,
            'class_distribution': class_counts,
            'alert_distribution': alert_counts,
            'confidence_stats': {
                'average': avg_confidence,
                'maximum': max_confidence,
                'minimum': min_confidence
            },
            'performance_stats': {
                'avg_inference_time_ms': avg_inference_time,
                'total_processing_time_s': processing_time,
                'fps_processed': total_frames / processing_time
            },
            'detailed_results': results
        }
        
        return summary
    
    def _print_analysis_summary(self, summary: Dict):
        """Print formatted analysis summary"""
        print("\n" + "="*60)
        print("VIDEO ANALYSIS SUMMARY")
        print("="*60)
        print(f"Frames Analyzed: {summary['total_frames_analyzed']}")
        print(f"Disaster Frames: {summary['disaster_frames_detected']} ({summary['disaster_percentage']:.1f}%)")
        print(f"Avg Inference Time: {summary['performance_stats']['avg_inference_time_ms']:.1f}ms")
        print(f"Average Confidence: {summary['confidence_stats']['average']:.1%}")
        
        print("\nCLASS DISTRIBUTION:")
        for cls, count in summary['class_distribution'].items():
            percentage = (count / summary['total_frames_analyzed']) * 100
            print(f"   {cls}: {count} frames ({percentage:.1f}%)")
        
        print("\nALERT LEVEL DISTRIBUTION:")
        for alert, count in summary['alert_distribution'].items():
            percentage = (count / summary['total_frames_analyzed']) * 100
            print(f"   {alert}: {count} frames ({percentage:.1f}%)")
        
        print("="*60)
    
    def save_analysis_report(self, summary: Dict, output_path: str):
        """Save detailed analysis report to JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"{output_path}/analysis_report_{timestamp}.json"
        
        with open(report_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"Analysis report saved: {report_path}")
        return report_path

def main():
    """Main function for custom video analysis"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Custom Video Disaster Analysis System')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--video_path', type=str, required=True,
                       help='Path to input video file')
    parser.add_argument('--output_path', type=str, default=None,
                       help='Path for output analyzed video')
    parser.add_argument('--config_path', type=str, default=None,
                       help='Path to model configuration file')
    parser.add_argument('--no_display', action='store_true',
                       help='Disable live video display')
    parser.add_argument('--save_report', action='store_true',
                       help='Save detailed analysis report')
    
    args = parser.parse_args()
    
    analyzer = CustomVideoDisasterAnalyzer(args.model_path, args.config_path)
    
    # Process video
    summary = analyzer.process_video(
        video_path=args.video_path,
        output_path=args.output_path,
        show_live=not args.no_display
    )
    
    if args.save_report:
        analyzer.save_analysis_report(summary, "outputs")
    
    print("🎬 Custom video analysis completed!")

if __name__ == "__main__":
    main() 