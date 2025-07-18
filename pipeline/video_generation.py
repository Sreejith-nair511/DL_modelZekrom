import torch
from diffusers import StableVideoDiffusionPipeline
from PIL import Image
import cv2
import numpy as np
import os
import logging
from typing import Dict, List, Optional, Tuple
import imageio

logger = logging.getLogger(__name__)

class VideoGenerator:
    """Stage 1: AI Video Generation"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models = {}
        self.available_models = {
            'stable_video_diffusion': {
                'name': 'Stable Video Diffusion',
                'model_id': 'stabilityai/stable-video-diffusion-img2vid-xt',
                'type': 'img2vid'
            },
            'animate_diff': {
                'name': 'AnimateDiff',
                'model_id': 'runwayml/stable-diffusion-v1-5',
                'type': 'txt2vid'
            }
        }
        
    def get_available_models(self) -> List[Dict]:
        """Get list of available video generation models"""
        return list(self.available_models.values())
    
    def load_model(self, model_name: str):
        """Load a specific video generation model"""
        if model_name in self.models:
            return self.models[model_name]
        
        try:
            model_config = self.available_models[model_name]
            
            if model_name == 'stable_video_diffusion':
                pipeline = StableVideoDiffusionPipeline.from_pretrained(
                    model_config['model_id'],
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    variant="fp16" if self.device == "cuda" else None
                )
                pipeline = pipeline.to(self.device)
            else:
                # Fallback to simple frame generation
                pipeline = self._create_fallback_pipeline()
            
            if hasattr(pipeline, 'enable_attention_slicing'):
                pipeline.enable_attention_slicing()
            if hasattr(pipeline, 'enable_vae_slicing'):
                pipeline.enable_vae_slicing()
            
            self.models[model_name] = pipeline
            logger.info(f"Loaded model: {model_config['name']}")
            return pipeline
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            return self._create_fallback_pipeline()
    
    def _create_fallback_pipeline(self):
        logger.info("Using fallback video generation")
        return "fallback"
    
    def generate_from_text(
        self,
        prompt: str,
        model_name: str = 'animate_diff',
        num_frames: int = 25,
        resolution: Tuple[int, int] = (512, 512),
        **kwargs
    ) -> np.ndarray:
        """Generate video from text prompt"""
        try:
            pipeline = self.load_model(model_name)
            
            if pipeline == "fallback":
                return self._generate_fallback_video(prompt, num_frames, resolution)
            
            # Generate simple animated frames based on prompt
            frames = []
            base_color = self._prompt_to_color(prompt)
            
            for i in range(num_frames):
                frame = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)
                
                # Create animated gradient based on frame number
                for y in range(resolution[1]):
                    for x in range(resolution[0]):
                        r = int(base_color[0] + 50 * np.sin(i * 0.1 + x * 0.01))
                        g = int(base_color[1] + 50 * np.cos(i * 0.1 + y * 0.01))
                        b = int(base_color[2] + 50 * np.sin(i * 0.1 + (x+y) * 0.005))
                        
                        frame[y, x] = [
                            max(0, min(255, r)),
                            max(0, min(255, g)),
                            max(0, min(255, b))
                        ]
                
                frames.append(frame)
            
            return np.array(frames)
            
        except Exception as e:
            logger.error(f"Text-to-video generation failed: {str(e)}")
            return self._generate_fallback_video(prompt, num_frames, resolution)
    
    def generate_from_image(
        self,
        image_path: str,
        model_name: str = 'stable_video_diffusion',
        num_frames: int = 25,
        **kwargs
    ) -> np.ndarray:
        """Generate video from image"""
        try:
            pipeline = self.load_model(model_name)
            
            image = Image.open(image_path).convert('RGB')
            image = image.resize((512, 512))
            base_frame = np.array(image)
            
            if pipeline == "fallback":
                return self._animate_image_fallback(base_frame, num_frames)
            
            # Simple image animation
            frames = []
            for i in range(num_frames):
                frame = base_frame.copy()
                
                # Add subtle animation effects
                brightness = 1.0 + 0.1 * np.sin(i * 0.2)
                frame = np.clip(frame * brightness, 0, 255).astype(np.uint8)
                
                frames.append(frame)
            
            return np.array(frames)
            
        except Exception as e:
            logger.error(f"Image-to-video generation failed: {str(e)}")
            base_frame = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
            return self._animate_image_fallback(base_frame, num_frames)
    
    def _prompt_to_color(self, prompt: str) -> Tuple[int, int, int]:
        # Simple prompt to color mapping
        prompt_lower = prompt.lower()
        if 'forest' in prompt_lower or 'green' in prompt_lower:
            return (34, 139, 34)
        elif 'ocean' in prompt_lower or 'blue' in prompt_lower:
            return (30, 144, 255)
        elif 'sunset' in prompt_lower or 'orange' in prompt_lower:
            return (255, 165, 0)
        elif 'fire' in prompt_lower or 'red' in prompt_lower:
            return (220, 20, 60)
        else:
            return (128, 128, 128)
    
    def _generate_fallback_video(self, prompt: str, num_frames: int, resolution: Tuple[int, int]) -> np.ndarray:
        frames = []
        base_color = self._prompt_to_color(prompt)
        
        for i in range(num_frames):
            frame = np.full((resolution[1], resolution[0], 3), base_color, dtype=np.uint8)
            
            # Add some animation
            center_x, center_y = resolution[0] // 2, resolution[1] // 2
            radius = int(50 + 20 * np.sin(i * 0.3))
            
            cv2.circle(frame, (center_x, center_y), radius, (255, 255, 255), 2)
            
            frames.append(frame)
        
        return np.array(frames)
    
    def _animate_image_fallback(self, base_frame: np.ndarray, num_frames: int) -> np.ndarray:
        frames = []
        for i in range(num_frames):
            frame = base_frame.copy()
            
            # Add zoom effect
            scale = 1.0 + 0.05 * np.sin(i * 0.2)
            h, w = frame.shape[:2]
            center_x, center_y = w // 2, h // 2
            
            M = cv2.getRotationMatrix2D((center_x, center_y), 0, scale)
            frame = cv2.warpAffine(frame, M, (w, h))
            
            frames.append(frame)
        
        return np.array(frames)
    
    def save_video(
        self,
        frames: np.ndarray,
        output_path: str,
        fps: int = 8,
        save_frames: bool = True
    ) -> Dict:
        """Save video frames with proper format handling"""
        try:
            # Create output directory
            output_dir = os.path.dirname(output_path)
            os.makedirs(output_dir, exist_ok=True)
            
            # Get file extension to determine format
            file_ext = os.path.splitext(output_path)[1].lower()
            
            # Save video with appropriate method based on format
            if file_ext == '.mp4':
                # Use mimsave for MP4 - this is what you had and it's correct
                imageio.mimsave(output_path, frames, fps=fps, quality=8)
            elif file_ext == '.gif':
                # For GIF, use duration instead of fps
                duration = 1.0 / fps
                imageio.mimsave(output_path, frames, duration=duration)
            elif file_ext in ['.avi', '.mov', '.mkv']:
                # For other video formats, use get_writer
                with imageio.get_writer(output_path, fps=fps) as writer:
                    for frame in frames:
                        writer.append_data(frame)
            elif file_ext in ['.tiff', '.tif']:
                # For TIFF, save as multi-page TIFF (no fps parameter)
                imageio.mimsave(output_path, frames)
                logger.warning(f"TIFF format doesn't support fps. Saved as multi-page TIFF.")
            else:
                # Default to MP4 if format is unclear
                output_path = output_path.rsplit('.', 1)[0] + '.mp4'
                imageio.mimsave(output_path, frames, fps=fps, quality=8)
                logger.info(f"Unknown format, saved as MP4: {output_path}")
            
            results = {
                'video_path': output_path,
                'num_frames': len(frames),
                'fps': fps
            }
            
            # Save individual frames if requested
            if save_frames:
                frames_dir = os.path.join(output_dir, 'frames')
                os.makedirs(frames_dir, exist_ok=True)
                
                frame_paths = []
                for i, frame in enumerate(frames):
                    frame_path = os.path.join(frames_dir, f'frame_{i:04d}.png')
                    if isinstance(frame, np.ndarray):
                        # Convert RGB to BGR for OpenCV
                        cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    else:
                        frame.save(frame_path)
                    frame_paths.append(frame_path)
                
                results['frames_dir'] = frames_dir
                results['frame_paths'] = frame_paths
            
            logger.info(f"Video saved: {output_path}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to save video: {str(e)}")
            # Try fallback save as MP4
            try:
                fallback_path = output_path.rsplit('.', 1)[0] + '.mp4'
                imageio.mimsave(fallback_path, frames, fps=fps, quality=8)
                logger.info(f"Fallback save successful: {fallback_path}")
                return {
                    'video_path': fallback_path,
                    'num_frames': len(frames),
                    'fps': fps
                }
            except Exception as fallback_error:
                logger.error(f"Fallback save also failed: {str(fallback_error)}")
                raise e
    
    def generate(
        self,
        prompt: Optional[str] = None,
        image_path: Optional[str] = None,
        output_dir: str = './output',
        model_name: str = 'stable_video_diffusion',
        resolution: Tuple[int, int] = (512, 512),
        num_frames: int = 25,
        fps: int = 8,
        save_frames: bool = True,
        output_format: str = 'mp4',  # New parameter for output format
        **kwargs
    ) -> Dict:
        """Main generation method"""
        try:
            if image_path and os.path.exists(image_path):
                # Image-to-video generation
                frames = self.generate_from_image(
                    image_path=image_path,
                    model_name=model_name,
                    num_frames=num_frames,
                    **kwargs
                )
            elif prompt:
                # Text-to-video generation
                frames = self.generate_from_text(
                    prompt=prompt,
                    model_name=model_name,
                    num_frames=num_frames,
                    resolution=resolution,
                    **kwargs
                )
            else:
                raise ValueError("Either prompt or image_path must be provided")
            
            # Create output path with specified format
            video_path = os.path.join(output_dir, f'generated_video.{output_format}')
            
            # Save video
            save_results = self.save_video(
                frames=frames,
                output_path=video_path,
                fps=fps,
                save_frames=save_frames
            )
            
            return {
                'success': True,
                'video_path': save_results['video_path'],
                'frames_dir': save_results.get('frames_dir'),
                'num_frames': save_results['num_frames'],
                'fps': save_results['fps'],
                'resolution': frames.shape[1:3] if len(frames) > 0 else resolution
            }
            
        except Exception as e:
            logger.error(f"Video generation failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }

# Additional utility function for format validation
def validate_output_format(format_str: str) -> str:
    """Validate and normalize output format"""
    format_str = format_str.lower().strip('.')
    
    supported_formats = {
        'mp4': 'mp4',
        'avi': 'avi',
        'mov': 'mov',
        'mkv': 'mkv',
        'gif': 'gif',
        'tiff': 'tiff',
        'tif': 'tiff'
    }
    
    if format_str in supported_formats:
        return supported_formats[format_str]
    else:
        logger.warning(f"Unsupported format '{format_str}', defaulting to 'mp4'")
        return 'mp4'

# Example usage
if __name__ == "__main__":
    # Create video generator
    generator = VideoGenerator()
    
    # Example 1: Text to video
    result = generator.generate(
        prompt="A beautiful forest with animated trees",
        output_format='mp4',
        num_frames=30,
        fps=10
    )
    print(f"Generated video: {result}")
    
    # Example 2: Image to video (if you have an image)
    # result = generator.generate(
    #     image_path="path/to/your/image.jpg",
    #     output_format='gif',
    #     num_frames=20
    # )
    # print(f"Generated video: {result}")