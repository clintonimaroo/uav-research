import os
import sys
from video_disaster_analyzer import CustomVideoDisasterAnalyzer

def main():
    print("🎬 Custom Video Disaster Analysis Demo")
    print("="*50)
    
    model_path = "../checkpoints/best_model.pth"  
    if not os.path.exists(model_path):
        print("❌ Model not found. Please train a model first or update the path.")
        print(f"   Looking for: {model_path}")
        return
    
    video_dir = "videos"
    if not os.path.exists(video_dir):
        print(f"📁 Creating {video_dir} directory for your videos...")
        os.makedirs(video_dir, exist_ok=True)
        print(f"   Please add your disaster videos to: {video_dir}/")
        return
    
    video_files = [f for f in os.listdir(video_dir) 
                  if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    
    if not video_files:
        print(f"❌ No video files found in {video_dir}/")
        print("   Supported formats: .mp4, .avi, .mov, .mkv")
        print("   Please add some test videos and try again.")
        return
    
    print(f"📹 Found {len(video_files)} video(s):")
    for i, video in enumerate(video_files):
        print(f"   {i+1}. {video}")
    
    try:
        choice = input(f"\nSelect video (1-{len(video_files)}): ")
        video_idx = int(choice) - 1
        
        if video_idx < 0 or video_idx >= len(video_files):
            print("❌ Invalid choice")
            return
            
        selected_video = video_files[video_idx]
        video_path = os.path.join(video_dir, selected_video)
        
    except (ValueError, KeyboardInterrupt):
        print("\n👋 Demo cancelled")
        return
    
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    output_video = os.path.join(output_dir, f"analyzed_{selected_video}")
    
    print(f"\nStarting analysis of: {selected_video}")
    print("Controls:")
    print("   - Press 'q' to quit")
    print("   - Press 'p' to pause/resume")
    print("   - Close window or Ctrl+C to stop")
    print()
    
    try:
        analyzer = CustomVideoDisasterAnalyzer(model_path)
        
        summary = analyzer.process_video(
            video_path=video_path,
            output_path=output_video,
            show_live=True  # Show real-time analysis
        )
        
        report_path = analyzer.save_analysis_report(summary, output_dir)
        
        print(f"\nAnalysis Complete!")
        print(f"Analyzed video saved: {output_video}")
        print(f"Detailed report saved: {report_path}")
        
    except Exception as e:
        print(f"\nError during analysis: {e}")
        print("Make sure your model is compatible and video format is supported.")

def quick_webcam_test():
    """Quick test using webcam"""
    print("🎥 Quick Webcam Test")
    print("="*30)
    
    model_path = "../checkpoints/best_model.pth"
    if not os.path.exists(model_path):
        print("Model not found")
        return
    
    try:
        import cv2
        
        # Test webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("No webcam found")
            return
        cap.release()
        
        print("📹 Using webcam for real-time analysis...")
        print("Press 'q' to quit")
        
        analyzer = CustomVideoDisasterAnalyzer(model_path)
        
        summary = analyzer.process_video(
            video_path=None, 
            output_path=None,  
            show_live=True
        )
        
        print("✅ Webcam test complete!")
        
    except Exception as e:
        print(f"❌ Webcam test failed: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "webcam":
        quick_webcam_test()
    else:
        main() 