import os
import json
import time
from datetime import datetime
from typing import List, Dict
from video_disaster_analyzer import CustomVideoDisasterAnalyzer

class BatchVideoProcessor:
    """Process multiple videos and generate comparative analysis"""
    
    def __init__(self, model_path: str, config_path: str = None):
        """Initialize batch processor with model"""
        print("🎬 Initializing Batch Video Processor...")
        
        self.analyzer = CustomVideoDisasterAnalyzer(model_path, config_path)
        self.batch_results = []
        
    def process_directory(self, videos_dir: str, output_dir: str = "batch_outputs") -> Dict:
        """Process all videos in a directory"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.webm')
        video_files = [f for f in os.listdir(videos_dir) 
                      if f.lower().endswith(video_extensions)]
        
        if not video_files:
            print(f"❌ No video files found in {videos_dir}")
            return {}
        
        print(f"📹 Found {len(video_files)} videos to process")
        
        batch_start = time.time()
        
        for i, video_file in enumerate(video_files, 1):
            print(f"\n{'='*60}")
            print(f"📹 Processing Video {i}/{len(video_files)}: {video_file}")
            print(f"{'='*60}")
            
            video_path = os.path.join(videos_dir, video_file)
            
            video_name = os.path.splitext(video_file)[0]
            output_video = os.path.join(output_dir, f"analyzed_{video_file}")
            
            try:
                # Process video
                summary = self.analyzer.process_video(
                    video_path=video_path,
                    output_path=output_video,
                    show_live=False 
                )
                
                summary['original_filename'] = video_file
                summary['output_filename'] = f"analyzed_{video_file}"
                summary['processing_order'] = i
                
                individual_report = os.path.join(output_dir, f"report_{video_name}.json")
                with open(individual_report, 'w') as f:
                    json.dump(summary, f, indent=2, default=str)
                
                self.batch_results.append(summary)
                
                print(f"✅ Completed: {video_file}")
                
            except Exception as e:
                print(f"❌ Error processing {video_file}: {e}")
                error_summary = {
                    'original_filename': video_file,
                    'error': str(e),
                    'processing_order': i,
                    'status': 'failed'
                }
                self.batch_results.append(error_summary)
        
        batch_time = time.time() - batch_start
        
        batch_analysis = self._generate_batch_analysis(batch_time)
        
        batch_report_path = os.path.join(output_dir, f"batch_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(batch_report_path, 'w') as f:
            json.dump(batch_analysis, f, indent=2, default=str)
        
        print(f"\n🎉 Batch processing complete!")
        print(f"📄 Batch report saved: {batch_report_path}")
        
        self._print_batch_summary(batch_analysis)
        
        return batch_analysis
    
    def _generate_batch_analysis(self, total_processing_time: float) -> Dict:
        """Generate comprehensive batch analysis"""
        
        successful_results = [r for r in self.batch_results if 'error' not in r]
        failed_results = [r for r in self.batch_results if 'error' in r]
        
        if not successful_results:
            return {
                'total_videos': len(self.batch_results),
                'successful': 0,
                'failed': len(failed_results),
                'total_processing_time': total_processing_time,
                'error': 'No videos processed successfully'
            }
        
        total_frames = sum(r['total_frames_analyzed'] for r in successful_results)
        total_disaster_frames = sum(r['disaster_frames_detected'] for r in successful_results)
        
        combined_classes = {}
        for result in successful_results:
            for cls, count in result['class_distribution'].items():
                combined_classes[cls] = combined_classes.get(cls, 0) + count
        
        combined_alerts = {}
        for result in successful_results:
            for alert, count in result['alert_distribution'].items():
                combined_alerts[alert] = combined_alerts.get(alert, 0) + count
        
        avg_inference_times = [r['performance_stats']['avg_inference_time_ms'] for r in successful_results]
        overall_avg_inference = sum(avg_inference_times) / len(avg_inference_times)
        
        video_summaries = []
        for result in successful_results:
            video_summaries.append({
                'filename': result['original_filename'],
                'frames_analyzed': result['total_frames_analyzed'],
                'disaster_percentage': result['disaster_percentage'],
                'dominant_class': max(result['class_distribution'].items(), key=lambda x: x[1])[0],
                'avg_confidence': result['confidence_stats']['average'],
                'processing_time': result['performance_stats']['total_processing_time_s']
            })
        
        batch_analysis = {
            'batch_summary': {
                'total_videos': len(self.batch_results),
                'successful': len(successful_results),
                'failed': len(failed_results),
                'total_processing_time': total_processing_time,
                'avg_processing_time_per_video': total_processing_time / len(self.batch_results)
            },
            'aggregate_analysis': {
                'total_frames_processed': total_frames,
                'total_disaster_frames': total_disaster_frames,
                'overall_disaster_percentage': (total_disaster_frames / total_frames * 100) if total_frames > 0 else 0,
                'combined_class_distribution': combined_classes,
                'combined_alert_distribution': combined_alerts,
                'overall_avg_inference_time_ms': overall_avg_inference
            },
            'video_summaries': video_summaries,
            'failed_videos': [{'filename': r['original_filename'], 'error': r['error']} for r in failed_results],
            'detailed_results': self.batch_results
        }
        
        return batch_analysis
    
    def _print_batch_summary(self, analysis: Dict):
        """Print formatted batch analysis summary"""
        print("\n" + "="*80)
        print("📊 BATCH ANALYSIS SUMMARY")
        print("="*80)
        
        batch = analysis['batch_summary']
        aggregate = analysis['aggregate_analysis']
        
        print(f"📹 Videos Processed: {batch['successful']}/{batch['total_videos']}")
        if batch['failed'] > 0:
            print(f"❌ Failed: {batch['failed']}")
        print(f"⏱️  Total Processing Time: {batch['total_processing_time']:.1f}s")
        print(f"📈 Total Frames Analyzed: {aggregate['total_frames_processed']:,}")
        print(f"🚨 Total Disaster Frames: {aggregate['total_disaster_frames']:,} ({aggregate['overall_disaster_percentage']:.1f}%)")
        print(f"⚡ Average Inference Time: {aggregate['overall_avg_inference_time_ms']:.1f}ms")
        
        print("\n📊 COMBINED CLASS DISTRIBUTION:")
        for cls, count in aggregate['combined_class_distribution'].items():
            percentage = (count / aggregate['total_frames_processed']) * 100
            print(f"   {cls}: {count:,} frames ({percentage:.1f}%)")
        
        print("\n⚠️ COMBINED ALERT DISTRIBUTION:")
        for alert, count in aggregate['combined_alert_distribution'].items():
            percentage = (count / aggregate['total_frames_processed']) * 100
            print(f"   {alert}: {count:,} frames ({percentage:.1f}%)")
        
        print("\n📹 PER-VIDEO RESULTS:")
        for video in analysis['video_summaries']:
            print(f"   📄 {video['filename']}")
            print(f"      Frames: {video['frames_analyzed']:,} | Disasters: {video['disaster_percentage']:.1f}%")
            print(f"      Dominant: {video['dominant_class']} | Confidence: {video['avg_confidence']:.1%}")
        
        if analysis['failed_videos']:
            print("\n❌ FAILED VIDEOS:")
            for failed in analysis['failed_videos']:
                print(f"   📄 {failed['filename']}: {failed['error']}")
        
        print("="*80)

def main():
    """Main function for batch processing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch Video Disaster Analysis')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--videos_dir', type=str, default='videos',
                       help='Directory containing videos to process')
    parser.add_argument('--output_dir', type=str, default='batch_outputs',
                       help='Directory for batch processing outputs')
    parser.add_argument('--config_path', type=str, default=None,
                       help='Path to model configuration file')
    
    args = parser.parse_args()
    
    processor = BatchVideoProcessor(args.model_path, args.config_path)
    
    results = processor.process_directory(args.videos_dir, args.output_dir)
    
    print("🎬 Batch processing completed!")

if __name__ == "__main__":
    main() 