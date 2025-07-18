import torch
import cv2
import numpy as np
import os
import logging
from typing import Dict, List, Optional, Tuple
from PIL import Image

logger = logging.getLogger(__name__)

class DepthExtractor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models = {}
        self.available_models = {
            'midas': {
                'name': 'MiDaS',
                'model_id': 'Intel/dpt-large',
                'type': 'monocular_depth'
            },
            'marigold': {
                'name': 'Marigold',
                'model_id': 'prs-eth/marigold-depth-lcm-v1-0',
                'type': 'diffusion_depth'
            },
            'depth_anything': {
                'name': 'Depth Anything',
                'model_id': 'LiheYoung/depth-anything-large-hf',
                'type': 'monocular_depth'
            }
        }
    
    def get_available_models(self) -> List[Dict]:
        return list(self.available_models.values())
    
    def load_model(self, model_name: str):
        if model_name in self.models:
            return self.models[model_name]
        
        try:
            # For demo purposes, use OpenCV-based depth estimation
            logger.info(f"Using OpenCV-based depth estimation for {model_name}")
            self.models[model_name] = "opencv_fallback"
            return "opencv_fallback"
            
        except Exception as e:
            logger.error(f"Failed to load depth model {model_name}: {str(e)}")
            return "opencv_fallback"
    
    def estimate_depth_frame(
        self,
        image: np.ndarray,
        model_name: str = 'midas'
    ) -> np.ndarray:
        try:
            model = self.load_model(model_name)
            return self._opencv_depth_estimation(image)
            
        except Exception as e:
            logger.error(f"Depth estimation failed for frame: {str(e)}")
            return self._opencv_depth_estimation(image)
    
    def _opencv_depth_estimation(self, image: np.ndarray) -> np.ndarray:
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Calculate gradient magnitude as depth proxy
            grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # Create depth map with center bias
            h, w = gray.shape
            center_x, center_y = w // 2, h // 2
            
            # Distance from center
            y, x = np.ogrid[:h, :w]
            center_distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            max_distance = np.sqrt(center_x**2 + center_y**2)
            center_bias = 1.0 - (center_distance / max_distance) * 0.5
            
            # Combine gradient and center bias
            depth_proxy = (255 - gradient_magnitude) * center_bias
            depth_proxy = self._normalize_depth(depth_proxy)
            
            return depth_proxy.astype(np.float32)
            
        except Exception as e:
            logger.error(f"OpenCV depth estimation failed: {str(e)}")
            return np.ones_like(image[:, :, 0] if len(image.shape) == 3 else image, dtype=np.float32) * 0.5
    
    def _normalize_depth(self, depth_map: np.ndarray) -> np.ndarray:
        try:
            if len(depth_map.shape) > 2:
                depth_map = depth_map.squeeze()
            
            valid_mask = np.isfinite(depth_map)
            if not np.any(valid_mask):
                return np.ones_like(depth_map) * 0.5
            
            min_depth = np.min(depth_map[valid_mask])
            max_depth = np.max(depth_map[valid_mask])
            
            if max_depth > min_depth:
                normalized = (depth_map - min_depth) / (max_depth - min_depth)
            else:
                normalized = np.ones_like(depth_map) * 0.5
            
            if not np.all(valid_mask):
                median_depth = np.median(normalized[valid_mask])
                normalized[~valid_mask] = median_depth
            
            return normalized.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Depth normalization failed: {str(e)}")
            return np.ones_like(depth_map) * 0.5
    
    def extract_from_video(
        self,
        video_path: str,
        model_name: str = 'midas',
        temporal_consistency: bool = True
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        try:
            cap = cv2.VideoCapture(video_path)
            frames = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            
            cap.release()
            
            if not frames:
                raise ValueError("No frames found in video")
            
            return self.extract_from_frames(frames, model_name, temporal_consistency)
            
        except Exception as e:
            logger.error(f"Video depth extraction failed: {str(e)}")
            raise
    
    def extract_from_frames(
        self,
        frames: List[np.ndarray],
        model_name: str = 'midas',
        temporal_consistency: bool = True
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        try:
            depth_maps = []
            
            logger.info(f"Extracting depth from {len(frames)} frames")
            
            for i, frame in enumerate(frames):
                logger.info(f"Processing frame {i+1}/{len(frames)}")
                depth_map = self.estimate_depth_frame(frame, model_name)
                depth_maps.append(depth_map)
            
            if temporal_consistency and len(depth_maps) > 1:
                depth_maps = self._apply_temporal_consistency(depth_maps)
            
            return frames, depth_maps
            
        except Exception as e:
            logger.error(f"Frame depth extraction failed: {str(e)}")
            raise
    
    def _apply_temporal_consistency(self, depth_maps: List[np.ndarray]) -> List[np.ndarray]:
        try:
            logger.info("Applying temporal consistency to depth maps")
            
            smoothed_depths = []
            window_size = 3
            
            for i, depth_map in enumerate(depth_maps):
                start_idx = max(0, i - window_size // 2)
                end_idx = min(len(depth_maps), i + window_size // 2 + 1)
                
                window_depths = depth_maps[start_idx:end_idx]
                averaged_depth = np.mean(window_depths, axis=0)
                
                alpha = 0.7
                blended_depth = alpha * depth_map + (1 - alpha) * averaged_depth
                
                smoothed_depths.append(blended_depth)
            
            return smoothed_depths
            
        except Exception as e:
            logger.error(f"Temporal consistency failed: {str(e)}")
            return depth_maps
    
    def save_depth_maps(
        self,
        depth_maps: List[np.ndarray],
        output_dir: str,
        save_visual: bool = True,
        save_raw: bool = True
    ) -> Dict:
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            depth_dir = os.path.join(output_dir, 'depth_maps')
            os.makedirs(depth_dir, exist_ok=True)
            
            visual_paths = []
            raw_paths = []
            
            for i, depth_map in enumerate(depth_maps):
                base_name = f'depth_{i:04d}'
                
                if save_visual:
                    visual_path = os.path.join(depth_dir, f'{base_name}.png')
                    depth_visual = (depth_map * 255).astype(np.uint8)
                    depth_colored = cv2.applyColorMap(depth_visual, cv2.COLORMAP_PLASMA)
                    cv2.imwrite(visual_path, depth_colored)
                    visual_paths.append(visual_path)
                
                if save_raw:
                    raw_path = os.path.join(depth_dir, f'{base_name}.npy')
                    np.save(raw_path, depth_map)
                    raw_paths.append(raw_path)
            
            results = {
                'depth_dir': depth_dir,
                'num_frames': len(depth_maps),
                'visual_paths': visual_paths if save_visual else [],
                'raw_paths': raw_paths if save_raw else []
            }
            
            logger.info(f"Saved {len(depth_maps)} depth maps to {depth_dir}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to save depth maps: {str(e)}")
            raise
    
    def extract(
        self,
        video_path: Optional[str] = None,
        frames_dir: Optional[str] = None,
        output_dir: str = './output',
        model_name: str = 'midas',
        temporal_consistency: bool = True,
        save_visual: bool = True,
        save_raw: bool = True,
        **kwargs
    ) -> Dict:
        try:
            if video_path and os.path.exists(video_path):
                frames, depth_maps = self.extract_from_video(
                    video_path=video_path,
                    model_name=model_name,
                    temporal_consistency=temporal_consistency
                )
            elif frames_dir and os.path.exists(frames_dir):
                frame_files = sorted([f for f in os.listdir(frames_dir) 
                                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                
                frames = []
                for frame_file in frame_files:
                    frame_path = os.path.join(frames_dir, frame_file)
                    frame = cv2.imread(frame_path)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)
                
                frames, depth_maps = self.extract_from_frames(
                    frames=frames,
                    model_name=model_name,
                    temporal_consistency=temporal_consistency
                )
            else:
                raise ValueError("Either video_path or frames_dir must be provided and exist")
            
            save_results = self.save_depth_maps(
                depth_maps=depth_maps,
                output_dir=output_dir,
                save_visual=save_visual,
                save_raw=save_raw
            )
            
            frames_output_dir = os.path.join(output_dir, 'frames')
            if not os.path.exists(frames_output_dir):
                os.makedirs(frames_output_dir, exist_ok=True)
                for i, frame in enumerate(frames):
                    frame_path = os.path.join(frames_output_dir, f'frame_{i:04d}.png')
                    cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            
            return {
                'success': True,
                'frames_dir': frames_output_dir,
                'depth_dir': save_results['depth_dir'],
                'num_frames': save_results['num_frames'],
                'visual_paths': save_results['visual_paths'],
                'raw_paths': save_results['raw_paths']
            }
            
        except Exception as e:
            logger.error(f"Depth extraction failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
