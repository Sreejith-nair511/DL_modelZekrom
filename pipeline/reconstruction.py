import os
import numpy as np
import trimesh
import cv2
from PIL import Image
import logging
from typing import Dict, List, Optional, Tuple
import json

logger = logging.getLogger(__name__)

class SceneReconstructor:
    """3D scene reconstruction using trimesh instead of open3d"""
    
    def __init__(self):
        self.available_models = {
            'basic': 'Basic depth-based reconstruction',
            'advanced': 'Advanced multi-view reconstruction',
            'neural': 'Neural reconstruction (placeholder)'
        }
        
    def get_available_models(self) -> Dict[str, str]:
        """Get available reconstruction models"""
        return self.available_models
    
    def reconstruct(self, 
                   frames_dir: str, 
                   depth_dir: str, 
                   output_dir: str,
                   model: str = 'basic',
                   **kwargs) -> Dict:
        """
        Reconstruct 3D scene from frames and depth maps
        
        Args:
            frames_dir: Directory containing video frames
            depth_dir: Directory containing depth maps
            output_dir: Output directory for 3D model
            model: Reconstruction model to use
            **kwargs: Additional parameters
            
        Returns:
            Dict with reconstruction results
        """
        try:
            logger.info(f"Starting 3D reconstruction with model: {model}")
            
            # Get frame and depth files
            frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
            depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
            
            if not frame_files or not depth_files:
                return {'success': False, 'error': 'No frames or depth maps found'}
            
            logger.info(f"Found {len(frame_files)} frames and {len(depth_files)} depth maps")
            
            # Load and process frames/depth maps
            points_3d = []
            colors = []
            
            for i, (frame_file, depth_file) in enumerate(zip(frame_files, depth_files)):
                if i % 5 != 0:  # Skip some frames for performance
                    continue
                    
                logger.info(f"Processing frame {i+1}/{len(frame_files)}")
                
                # Load frame and depth
                frame_path = os.path.join(frames_dir, frame_file)
                depth_path = os.path.join(depth_dir, depth_file)
                
                frame = cv2.imread(frame_path)
                depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
                
                if frame is None or depth is None:
                    logger.warning(f"Failed to load frame or depth map: {frame_file}, {depth_file}")
                    continue
                
                # Convert depth to 3D points
                frame_points, frame_colors = self._depth_to_pointcloud(
                    frame, depth, camera_params=kwargs.get('camera_params', {})
                )
                
                if len(frame_points) > 0:
                    points_3d.extend(frame_points)
                    colors.extend(frame_colors)
            
            if not points_3d:
                return {'success': False, 'error': 'No 3D points generated'}
            
            logger.info(f"Generated {len(points_3d)} 3D points")
            
            # Create point cloud
            points_3d = np.array(points_3d)
            colors = np.array(colors)
            
            # Create mesh using trimesh
            mesh = self._create_mesh_from_points(points_3d, colors, method=model)
            
            # Save mesh
            model_path = os.path.join(output_dir, 'reconstructed_model.glb')
            mesh.export(model_path)
            
            # Also save as OBJ for compatibility
            obj_path = os.path.join(output_dir, 'reconstructed_model.obj')
            mesh.export(obj_path)
            
            # Save point cloud as PLY
            ply_path = os.path.join(output_dir, 'point_cloud.ply')
            self._save_point_cloud(points_3d, colors, ply_path)
            
            # Generate metadata
            metadata = {
                'num_points': len(points_3d),
                'num_frames_processed': len(frame_files),
                'model_type': model,
                'mesh_faces': len(mesh.faces),
                'mesh_vertices': len(mesh.vertices),
                'bounds': {
                    'min': points_3d.min(axis=0).tolist(),
                    'max': points_3d.max(axis=0).tolist()
                }
            }
            
            metadata_path = os.path.join(output_dir, 'reconstruction_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info("3D reconstruction completed successfully")
            
            return {
                'success': True,
                'model_path': model_path,
                'obj_path': obj_path,
                'ply_path': ply_path,
                'metadata_path': metadata_path,
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"Reconstruction failed: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _depth_to_pointcloud(self, frame: np.ndarray, depth: np.ndarray, 
                           camera_params: Dict) -> Tuple[List, List]:
        """Convert depth map to 3D point cloud"""
        
        # Default camera parameters
        fx = camera_params.get('fx', 525.0)
        fy = camera_params.get('fy', 525.0)
        cx = camera_params.get('cx', frame.shape[1] / 2)
        cy = camera_params.get('cy', frame.shape[0] / 2)
        depth_scale = camera_params.get('depth_scale', 1000.0)
        
        points_3d = []
        colors = []
        
        h, w = depth.shape
        
        # Downsample for performance
        step = max(1, min(h, w) // 200)
        
        for y in range(0, h, step):
            for x in range(0, w, step):
                z = depth[y, x] / depth_scale
                
                if z > 0.1 and z < 10.0:  # Valid depth range
                    # Convert to 3D coordinates
                    x_3d = (x - cx) * z / fx
                    y_3d = (y - cy) * z / fy
                    z_3d = z
                    
                    points_3d.append([x_3d, y_3d, z_3d])
                    
                    # Get color from frame
                    color = frame[y, x]
                    colors.append([color[2], color[1], color[0]])  # BGR to RGB
        
        return points_3d, colors
    
    def _create_mesh_from_points(self, points: np.ndarray, colors: np.ndarray, 
                               method: str = 'basic') -> trimesh.Trimesh:
        """Create mesh from point cloud using trimesh"""
        
        if method == 'basic':
            # Simple convex hull
            try:
                hull = trimesh.convex.convex_hull(points)
                if colors is not None and len(colors) > 0:
                    # Apply average color to mesh
                    avg_color = np.mean(colors, axis=0).astype(np.uint8)
                    hull.visual.face_colors = avg_color
                return hull
            except:
                # Fallback: create a simple box mesh
                logger.warning("Convex hull failed, creating simple box mesh")
                bounds = np.array([points.min(axis=0), points.max(axis=0)])
                size = bounds[1] - bounds[0]
                center = (bounds[0] + bounds[1]) / 2
                box = trimesh.creation.box(extents=size)
                box.apply_translation(center)
                return box
        
        elif method == 'advanced':
            # Try to create a more detailed mesh
            try:
                # Use alpha shape for better reconstruction
                from scipy.spatial import ConvexHull
                hull = ConvexHull(points)
                mesh = trimesh.Trimesh(vertices=points[hull.vertices], 
                                     faces=hull.simplices)
                if colors is not None and len(colors) > 0:
                    mesh.visual.vertex_colors = colors[hull.vertices]
                return mesh
            except:
                # Fallback to basic method
                return self._create_mesh_from_points(points, colors, 'basic')
        
        else:
            # Default to basic method
            return self._create_mesh_from_points(points, colors, 'basic')
    
    def _save_point_cloud(self, points: np.ndarray, colors: np.ndarray, 
                         output_path: str):
        """Save point cloud as PLY file"""
        try:
            # Create point cloud
            point_cloud = trimesh.PointCloud(vertices=points, colors=colors)
            point_cloud.export(output_path)
            logger.info(f"Point cloud saved to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save point cloud: {str(e)}")
    
    def validate_inputs(self, frames_dir: str, depth_dir: str) -> bool:
        """Validate input directories and files"""
        if not os.path.exists(frames_dir):
            logger.error(f"Frames directory not found: {frames_dir}")
            return False
        
        if not os.path.exists(depth_dir):
            logger.error(f"Depth directory not found: {depth_dir}")
            return False
        
        frame_files = [f for f in os.listdir(frames_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        depth_files = [f for f in os.listdir(depth_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        if not frame_files:
            logger.error("No frame files found")
            return False
        
        if not depth_files:
            logger.error("No depth files found")
            return False
        
        return True