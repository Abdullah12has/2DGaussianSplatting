#
# Monocular Depth and Normal Prior Estimation
# For 2D Gaussian Splatting enhancement
#

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path
import os

# Global model instances (lazy-loaded)
_depth_model = None
_normal_model = None
_depth_processor = None


def get_depth_model(device="cuda"):
    """
    Load Depth-Anything-V2 model for monocular depth estimation.
    Uses HuggingFace transformers.
    """
    global _depth_model, _depth_processor
    
    if _depth_model is None:
        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
            
            model_name = "depth-anything/Depth-Anything-V2-Small-hf"
            print(f"Loading depth model: {model_name}")
            
            _depth_processor = AutoImageProcessor.from_pretrained(model_name)
            _depth_model = AutoModelForDepthEstimation.from_pretrained(model_name)
            _depth_model = _depth_model.to(device)
            _depth_model.eval()
            
        except ImportError:
            print("Warning: transformers not installed. Install with: pip install transformers")
            print("Falling back to Omnidata depth model...")
            return get_omnidata_depth_model(device)
    
    return _depth_model, _depth_processor


def get_omnidata_depth_model(device="cuda"):
    """
    Fallback: Load Omnidata depth model.
    """
    # Placeholder for Omnidata implementation
    raise NotImplementedError(
        "Omnidata depth model not yet implemented. "
        "Please install transformers and use Depth-Anything-V2."
    )


def get_normal_model(device="cuda"):
    """
    Load Omnidata normal estimation model.
    
    Omnidata models should be downloaded from:
    https://github.com/EPFL-VILAB/omnidata
    
    Place weights in: ./pretrained_models/omnidata_dpt_normal_v2.ckpt
    """
    global _normal_model
    
    if _normal_model is None:
        try:
            from omnidata_tools.model_util import load_omnidata_normal_model
            
            weights_path = Path(__file__).parent.parent / "pretrained_models" / "omnidata_dpt_normal_v2.ckpt"
            
            if not weights_path.exists():
                print(f"Warning: Omnidata normal weights not found at {weights_path}")
                print("Please download from: https://github.com/EPFL-VILAB/omnidata")
                return None
            
            _normal_model = load_omnidata_normal_model(weights_path, device)
            _normal_model.eval()
            
        except ImportError:
            print("Warning: omnidata_tools not installed.")
            print("Using simple Sobel-based normal estimation as fallback.")
            return None
    
    return _normal_model


def estimate_depth(image, device="cuda"):
    """
    Estimate monocular depth from an image.
    
    Args:
        image: PIL Image or numpy array [H, W, 3] (0-255)
        device: torch device
        
    Returns:
        depth: numpy array [H, W] with relative depth values
    """
    model, processor = get_depth_model(device)
    
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype(np.uint8))
    
    # Process image
    inputs = processor(images=image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth
    
    # Interpolate to original size
    prediction = F.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],  # PIL size is (W, H)
        mode="bicubic",
        align_corners=False,
    )
    
    depth = prediction.squeeze().cpu().numpy()
    return depth


def estimate_normal_from_depth(depth, mask=None):
    """
    Estimate surface normals from depth map using Sobel gradients.
    Fallback when Omnidata is not available.
    
    Args:
        depth: numpy array [H, W]
        mask: optional valid region mask
        
    Returns:
        normal: numpy array [H, W, 3] with xyz normal components
    """
    # Compute gradients
    dz_dx = np.gradient(depth, axis=1)
    dz_dy = np.gradient(depth, axis=0)
    
    # Construct normal vectors
    # Normal = (-dz/dx, -dz/dy, 1), normalized
    normal = np.stack([-dz_dx, -dz_dy, np.ones_like(depth)], axis=-1)
    
    # Normalize
    norm = np.linalg.norm(normal, axis=-1, keepdims=True)
    normal = normal / (norm + 1e-8)
    
    return normal


def estimate_normal(image, device="cuda"):
    """
    Estimate surface normals from an image.
    
    Args:
        image: PIL Image or numpy array [H, W, 3] (0-255)
        device: torch device
        
    Returns:
        normal: numpy array [H, W, 3] with xyz normal components in [-1, 1]
    """
    model = get_normal_model(device)
    
    if model is None:
        # Fallback: estimate from depth
        depth = estimate_depth(image, device)
        return estimate_normal_from_depth(depth)
    
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype(np.uint8))
    
    # Prepare input for Omnidata
    from torchvision import transforms
    
    transform = transforms.Compose([
        transforms.Resize(384),
        transforms.CenterCrop(384),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        normal_pred = model(input_tensor)
    
    # Resize to original
    normal_pred = F.interpolate(
        normal_pred,
        size=image.size[::-1],
        mode="bilinear",
        align_corners=False,
    )
    
    # Convert to numpy, map from [0,1] to [-1,1]
    normal = normal_pred.squeeze().permute(1, 2, 0).cpu().numpy()
    normal = normal * 2 - 1
    
    # Normalize
    norm = np.linalg.norm(normal, axis=-1, keepdims=True)
    normal = normal / (norm + 1e-8)
    
    return normal


class MonoPriorProcessor:
    """
    Process and cache monocular priors for a dataset.
    """
    
    def __init__(self, device="cuda"):
        self.device = device
    
    def compute_priors(self, image_paths, output_dir, force_recompute=False):
        """
        Compute and save depth/normal priors for a list of images.
        
        Args:
            image_paths: List of image file paths
            output_dir: Directory to save priors (creates mono_depth/, mono_normal/ subdirs)
            force_recompute: If True, recompute even if files exist
        """
        output_dir = Path(output_dir)
        depth_dir = output_dir / "mono_depth"
        normal_dir = output_dir / "mono_normal"
        
        depth_dir.mkdir(parents=True, exist_ok=True)
        normal_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Computing monocular priors for {len(image_paths)} images...")
        
        for img_path in image_paths:
            img_path = Path(img_path)
            img_name = img_path.stem
            
            depth_path = depth_dir / f"{img_name}.npy"
            normal_path = normal_dir / f"{img_name}.npy"
            
            # Skip if exists and not forcing recompute
            if not force_recompute and depth_path.exists() and normal_path.exists():
                continue
            
            # Load image
            image = Image.open(img_path).convert("RGB")
            
            # Compute depth
            if not depth_path.exists() or force_recompute:
                depth = estimate_depth(image, self.device)
                np.save(depth_path, depth.astype(np.float32))
            
            # Compute normal
            if not normal_path.exists() or force_recompute:
                normal = estimate_normal(image, self.device)
                np.save(normal_path, normal.astype(np.float32))
        
        print(f"Priors saved to: {output_dir}")
    
    @staticmethod
    def load_depth(path):
        """Load cached depth prior."""
        if Path(path).exists():
            return np.load(path)
        return None
    
    @staticmethod
    def load_normal(path):
        """Load cached normal prior."""
        if Path(path).exists():
            return np.load(path)
        return None
    
    @staticmethod
    def get_prior_paths(image_path, prior_dir):
        """
        Get paths to depth/normal priors for an image.
        
        Args:
            image_path: Path to the original image
            prior_dir: Base directory containing mono_depth/ and mono_normal/
            
        Returns:
            (depth_path, normal_path) tuple
        """
        prior_dir = Path(prior_dir)
        img_name = Path(image_path).stem
        
        depth_path = prior_dir / "mono_depth" / f"{img_name}.npy"
        normal_path = prior_dir / "mono_normal" / f"{img_name}.npy"
        
        return depth_path, normal_path
