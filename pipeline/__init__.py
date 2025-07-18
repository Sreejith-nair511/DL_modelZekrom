"""
AI-to-3D VR Pipeline

A comprehensive pipeline for transforming text or image prompts into interactive VR/AR experiences.

Stages:
1. Video Generation - Generate AI videos from prompts
2. Depth Extraction - Extract depth maps from video frames  
3. 3D Reconstruction - Reconstruct 3D scenes using RGB + Depth
4. Game Engine Export - Import assets into Unity/Unreal
5. VR Deployment - Package for VR/AR platforms

Author: AI-to-3D VR Pipeline Team
Version: 1.0.0
"""

from .video_generation import VideoGenerator
from .depth_extraction import DepthExtractor
from .reconstruction import SceneReconstructor
from .exporter import GameEngineExporter
from .deploy_vr import VRDeployer

__version__ = "1.0.0"
__author__ = "AI-to-3D VR Pipeline Team"

__all__ = [
    'VideoGenerator',
    'DepthExtractor', 
    'SceneReconstructor',
    'GameEngineExporter',
    'VRDeployer'
]

# Pipeline stage information
PIPELINE_STAGES = {
    1: {
        'name': 'Video Generation',
        'class': VideoGenerator,
        'description': 'Generate AI videos from text prompts or images'
    },
    2: {
        'name': 'Depth Extraction', 
        'class': DepthExtractor,
        'description': 'Extract per-frame depth maps from video'
    },
    3: {
        'name': '3D Reconstruction',
        'class': SceneReconstructor, 
        'description': 'Reconstruct 3D scenes from RGB + Depth data'
    },
    4: {
        'name': 'Game Engine Export',
        'class': GameEngineExporter,
        'description': 'Export 3D assets to Unity/Unreal Engine'
    },
    5: {
        'name': 'VR Deployment',
        'class': VRDeployer,
        'description': 'Deploy to VR/AR platforms (Quest, Vision Pro, WebXR)'
    }
}

def get_pipeline_info():
    """Get information about all pipeline stages"""
    return PIPELINE_STAGES

def create_pipeline():
    """Create instances of all pipeline stages"""
    return {
        'video_generator': VideoGenerator(),
        'depth_extractor': DepthExtractor(),
        'scene_reconstructor': SceneReconstructor(),
        'game_engine_exporter': GameEngineExporter(),
        'vr_deployer': VRDeployer()
    }
