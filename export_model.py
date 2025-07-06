import os
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np
import argparse
from typing import Tuple, Optional

from config import TrainingConfig
from models import create_model
from utils import load_checkpoint

class ModelExporter:
    """Export trained models for deployment"""
    
    def __init__(self, model_path: str, config_path: str = None):
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load checkpoint
        self.checkpoint = torch.load(model_path, map_location=self.device)
        
        # Load configuration
        if config_path:
            self.config = TrainingConfig.load_config(config_path)
        else:
            self.config = self.checkpoint.get('config', None)
            if self.config is None:
                raise ValueError("No configuration found. Please provide config_path.")
        
        self.class_to_idx = self.checkpoint.get('class_to_idx', {})
        
        # Initialize model
        self.model = create_model(
            model_name=self.config.model_name,
            num_classes=self.config.num_classes
        )
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Loaded model: {self.config.model_name}")
        print(f"Classes: {list(self.class_to_idx.keys())}")
        print(f"Input size: {self.config.input_size}")
    
    def export_torchscript(self, save_path: str, input_size: Optional[Tuple[int, int, int]] = None):
        """Export model to TorchScript format"""
        if input_size is None:
            input_size = (3, self.config.input_size, self.config.input_size)
        
        print(f"Exporting to TorchScript: {save_path}")
        
        # Create dummy input
        dummy_input = torch.randn(1, *input_size)
        
        # Trace the model
        try:
            traced_model = torch.jit.trace(self.model, dummy_input)
            
            # Save the traced model
            traced_model.save(save_path)
            
            # Verify the export
            self._verify_torchscript_export(save_path, dummy_input)
            
            print(f"✓ TorchScript export successful: {save_path}")
            return True
            
        except Exception as e:
            print(f"✗ TorchScript export failed: {e}")
            return False
    
    def export_onnx(self, save_path: str, input_size: Optional[Tuple[int, int, int]] = None,
                   opset_version: int = 11):
        """Export model to ONNX format"""
        if input_size is None:
            input_size = (3, self.config.input_size, self.config.input_size)
        
        print(f"Exporting to ONNX: {save_path}")
        
        # Create dummy input
        dummy_input = torch.randn(1, *input_size)
        
        # Define input and output names
        input_names = ['input']
        output_names = ['output']
        
        # Define dynamic axes for variable batch size
        dynamic_axes = {
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
        
        try:
            # Export to ONNX
            torch.onnx.export(
                self.model,
                dummy_input,
                save_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes
            )
            
            # Verify the export
            self._verify_onnx_export(save_path, dummy_input)
            
            print(f"✓ ONNX export successful: {save_path}")
            return True
            
        except Exception as e:
            print(f"✗ ONNX export failed: {e}")
            return False
    
    def _verify_torchscript_export(self, model_path: str, dummy_input: torch.Tensor):
        """Verify TorchScript export by comparing outputs"""
        # Load the exported model
        loaded_model = torch.jit.load(model_path)
        loaded_model.eval()
        
        # Compare outputs
        with torch.no_grad():
            original_output = self.model(dummy_input)
            exported_output = loaded_model(dummy_input)
            
            # Check if outputs are close
            if torch.allclose(original_output, exported_output, atol=1e-5):
                print("  ✓ TorchScript verification passed")
            else:
                print("  ⚠️ TorchScript verification failed - outputs differ")
                print(f"    Max difference: {torch.max(torch.abs(original_output - exported_output))}")
    
    def _verify_onnx_export(self, model_path: str, dummy_input: torch.Tensor):
        """Verify ONNX export by comparing outputs"""
        try:
            # Load ONNX model
            onnx_model = onnx.load(model_path)
            onnx.checker.check_model(onnx_model)
            
            # Create ONNX Runtime session
            ort_session = ort.InferenceSession(model_path)
            
            # Get input name
            input_name = ort_session.get_inputs()[0].name
            
            # Run inference
            with torch.no_grad():
                original_output = self.model(dummy_input).numpy()
                onnx_output = ort_session.run(None, {input_name: dummy_input.numpy()})[0]
                
                # Check if outputs are close
                if np.allclose(original_output, onnx_output, atol=1e-5):
                    print("  ✓ ONNX verification passed")
                else:
                    print("  ⚠️ ONNX verification failed - outputs differ")
                    print(f"    Max difference: {np.max(np.abs(original_output - onnx_output))}")
                    
        except Exception as e:
            print(f"  ⚠️ ONNX verification failed: {e}")
    
    def export_model_info(self, save_path: str):
        """Export model metadata and class information"""
        model_info = {
            'model_name': self.config.model_name,
            'num_classes': self.config.num_classes,
            'input_size': self.config.input_size,
            'classes': list(self.class_to_idx.keys()),
            'class_to_idx': self.class_to_idx,
            'best_val_acc': self.checkpoint.get('best_val_acc', None),
            'normalization': {
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225]
            }
        }
        
        # Add model-specific info
        if hasattr(self.model, 'get_model_info'):
            model_stats = self.model.get_model_info()
            model_info.update(model_stats)
        
        # Save to JSON
        import json
        with open(save_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print(f"✓ Model info saved: {save_path}")
    
    def benchmark_inference(self, num_runs: int = 100, batch_size: int = 1):
        """Benchmark inference speed"""
        input_size = (3, self.config.input_size, self.config.input_size)
        dummy_input = torch.randn(batch_size, *input_size)
        
        # Warm up
        with torch.no_grad():
            for _ in range(10):
                _ = self.model(dummy_input)
        
        # Benchmark
        import time
        times = []
        
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                _ = self.model(dummy_input)
                end_time = time.time()
                times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        fps = batch_size / avg_time
        
        print(f"\nInference Benchmark (batch_size={batch_size}):")
        print(f"  Average time: {avg_time*1000:.2f} ± {std_time*1000:.2f} ms")
        print(f"  FPS: {fps:.1f}")
        print(f"  Throughput: {fps * batch_size:.1f} images/sec")
        
        return avg_time, fps
    
    def export_all(self, export_dir: str):
        """Export model in all formats"""
        os.makedirs(export_dir, exist_ok=True)
        
        model_name = f"{self.config.model_name}_{self.config.num_classes}classes"
        
        # Export TorchScript
        torchscript_path = os.path.join(export_dir, f"{model_name}.pt")
        self.export_torchscript(torchscript_path)
        
        # Export ONNX
        onnx_path = os.path.join(export_dir, f"{model_name}.onnx")
        self.export_onnx(onnx_path)
        
        # Export model info
        info_path = os.path.join(export_dir, f"{model_name}_info.json")
        self.export_model_info(info_path)
        
        # Benchmark
        self.benchmark_inference()
        
        print(f"\n✓ All exports completed in: {export_dir}")
        
        # Create deployment README
        self._create_deployment_readme(export_dir, model_name)
    
    def _create_deployment_readme(self, export_dir: str, model_name: str):
        """Create deployment instructions"""
        readme_content = f"""# Disaster Detection Model Deployment

## Model Information
- **Model**: {self.config.model_name}
- **Classes**: {list(self.class_to_idx.keys())}
- **Input Size**: {self.config.input_size}x{self.config.input_size}
- **Best Validation Accuracy**: {self.checkpoint.get('best_val_acc', 'N/A'):.2f}%

## Files
- `{model_name}.pt` - TorchScript model for PyTorch deployment
- `{model_name}.onnx` - ONNX model for cross-platform deployment
- `{model_name}_info.json` - Model metadata and class information

## Usage

### PyTorch/TorchScript
```python
import torch
from torchvision import transforms

# Load model
model = torch.jit.load('{model_name}.pt')
model.eval()

# Preprocessing
transform = transforms.Compose([
    transforms.Resize(({self.config.input_size}, {self.config.input_size})),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

# Inference
with torch.no_grad():
    input_tensor = transform(image).unsqueeze(0)
    output = model(input_tensor)
    prediction = torch.softmax(output, dim=1)
```

### ONNX Runtime
```python
import onnxruntime as ort
import numpy as np

# Load model
session = ort.InferenceSession('{model_name}.onnx')

# Inference
input_name = session.get_inputs()[0].name
output = session.run(None, {{input_name: input_array}})
```

## Deployment Notes
- **Input**: RGB images, normalized with ImageNet statistics
- **Output**: Logits (apply softmax for probabilities)
- **Classes**: {dict(self.class_to_idx)}

## UAV Integration
This model is optimized for UAV deployment with:
- Lightweight architecture ({self.config.model_name})
- Efficient inference
- Standard input/output formats

For real-time UAV deployment, consider:
1. Using TensorRT for NVIDIA hardware
2. Quantization for further optimization
3. Batch processing for multiple frames
"""
        
        readme_path = os.path.join(export_dir, 'README.md')
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        
        print(f"✓ Deployment README created: {readme_path}")

def main():
    """Main export function"""
    parser = argparse.ArgumentParser(description='Export disaster detection model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config_path', type=str, default=None,
                       help='Path to configuration file')
    parser.add_argument('--export_dir', type=str, default='exported_models',
                       help='Directory to save exported models')
    parser.add_argument('--format', type=str, choices=['torchscript', 'onnx', 'all'],
                       default='all', help='Export format')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run inference benchmark')
    
    args = parser.parse_args()
    
    # Create exporter
    exporter = ModelExporter(args.model_path, args.config_path)
    
    # Export based on format
    if args.format == 'all':
        exporter.export_all(args.export_dir)
    elif args.format == 'torchscript':
        os.makedirs(args.export_dir, exist_ok=True)
        model_name = f"{exporter.config.model_name}_{exporter.config.num_classes}classes"
        torchscript_path = os.path.join(args.export_dir, f"{model_name}.pt")
        exporter.export_torchscript(torchscript_path)
    elif args.format == 'onnx':
        os.makedirs(args.export_dir, exist_ok=True)
        model_name = f"{exporter.config.model_name}_{exporter.config.num_classes}classes"
        onnx_path = os.path.join(args.export_dir, f"{model_name}.onnx")
        exporter.export_onnx(onnx_path)
    
    # Run benchmark if requested
    if args.benchmark:
        exporter.benchmark_inference()
    
    print(f"\nExport completed!")

if __name__ == "__main__":
    main() 