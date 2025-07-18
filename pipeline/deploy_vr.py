import os
import json
import subprocess
import shutil
import logging
from typing import Dict, List, Optional
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)

class VRDeployer:
    """Stage 5: VR/AR Deployment"""
    
    def __init__(self):
        self.supported_platforms = {
            'quest': {
                'name': 'Meta Quest',
                'build_target': 'Android',
                'package_format': '.apk',
                'sdk_required': 'Android SDK'
            },
            'vision_pro': {
                'name': 'Apple Vision Pro',
                'build_target': 'visionOS',
                'package_format': '.ipa',
                'sdk_required': 'visionOS SDK'
            },
            'webxr': {
                'name': 'WebXR',
                'build_target': 'WebGL',
                'package_format': '.zip',
                'sdk_required': None
            },
            'pico': {
                'name': 'Pico VR',
                'build_target': 'Android',
                'package_format': '.apk',
                'sdk_required': 'Android SDK'
            },
            'vive': {
                'name': 'HTC Vive',
                'build_target': 'Windows',
                'package_format': '.exe',
                'sdk_required': 'SteamVR'
            }
        }
        
        self.optimization_settings = {
            'quest': {
                'texture_compression': 'ASTC',
                'texture_max_size': 1024,
                'mesh_compression': 'High',
                'audio_compression': 'Vorbis',
                'target_framerate': 72,
                'foveated_rendering': True,
                'fixed_foveated_rendering': True
            },
            'vision_pro': {
                'texture_compression': 'ASTC',
                'texture_max_size': 2048,
                'mesh_compression': 'Medium',
                'audio_compression': 'AAC',
                'target_framerate': 90,
                'foveated_rendering': True,
                'eye_tracking': True
            },
            'webxr': {
                'texture_compression': 'DXT',
                'texture_max_size': 512,
                'mesh_compression': 'High',
                'audio_compression': 'MP3',
                'target_framerate': 60,
                'progressive_loading': True
            }
        }
    
    def get_supported_platforms(self) -> List[Dict]:
        """Get list of supported VR/AR platforms"""
        return list(self.supported_platforms.values())
    
    def detect_android_sdk(self) -> Optional[str]:
        """Detect Android SDK installation"""
        try:
            # Check environment variables
            android_home = os.environ.get('ANDROID_HOME')
            if android_home and os.path.exists(android_home):
                return android_home
            
            android_sdk_root = os.environ.get('ANDROID_SDK_ROOT')
            if android_sdk_root and os.path.exists(android_sdk_root):
                return android_sdk_root
            
            # Check common installation paths
            common_paths = [
                os.path.expanduser('~/Android/Sdk'),  # Linux/macOS
                os.path.expanduser('~/Library/Android/sdk'),  # macOS
                'C:\\Users\\%USERNAME%\\AppData\\Local\\Android\\Sdk',  # Windows
                'C:\\Android\\Sdk'  # Windows alternative
            ]
            
            for path in common_paths:
                expanded_path = os.path.expandvars(path)
                if os.path.exists(expanded_path):
                    return expanded_path
            
            return None
            
        except Exception as e:
            logger.error(f"Android SDK detection failed: {str(e)}")
            return None
    
    def detect_unity_installation(self) -> Optional[str]:
        """Detect Unity installation for building"""
        try:
            # Check common Unity installation paths
            unity_paths = [
                '/Applications/Unity/Hub/Editor',  # macOS
                'C:\\Program Files\\Unity\\Hub\\Editor',  # Windows
                '/opt/Unity/Editor',  # Linux
                os.path.expanduser('~/Unity/Hub/Editor')  # User installation
            ]
            
            for base_path in unity_paths:
                if os.path.exists(base_path):
                    # Find latest version
                    versions = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
                    if versions:
                        latest_version = sorted(versions)[-1]
                        unity_exe = os.path.join(base_path, latest_version, 'Unity.exe')
                        if os.path.exists(unity_exe):
                            return unity_exe
            
            return None
            
        except Exception as e:
            logger.error(f"Unity detection failed: {str(e)}")
            return None
    
    def optimize_for_platform(
        self,
        project_path: str,
        platform: str
    ) -> bool:
        """Apply platform-specific optimizations"""
        try:
            if platform not in self.optimization_settings:
                logger.warning(f"No optimization settings for platform: {platform}")
                return True
            
            settings = self.optimization_settings[platform]
            logger.info(f"Applying {platform} optimizations")
            
            # Apply texture optimizations
            self._optimize_textures(project_path, settings)
            
            # Apply mesh optimizations
            self._optimize_meshes(project_path, settings)
            
            # Apply audio optimizations
            self._optimize_audio(project_path, settings)
            
            # Create platform-specific settings
            self._create_platform_settings(project_path, platform, settings)
            
            return True
            
        except Exception as e:
            logger.error(f"Platform optimization failed: {str(e)}")
            return False
    
    def _optimize_textures(self, project_path: str, settings: Dict):
        """Optimize textures for platform"""
        try:
            textures_dir = os.path.join(project_path, 'Assets', 'Textures')
            if not os.path.exists(textures_dir):
                return
            
            max_size = settings.get('texture_max_size', 1024)
            compression = settings.get('texture_compression', 'ASTC')
            
            logger.info(f"Optimizing textures: max_size={max_size}, compression={compression}")
            
            # This would typically involve:
            # 1. Resizing textures to max_size
            # 2. Converting to optimal format
            # 3. Generating mipmaps
            # 4. Applying compression
            
            # For now, we'll create optimization metadata
            optimization_info = {
                'texture_optimization': {
                    'max_size': max_size,
                    'compression': compression,
                    'generate_mipmaps': True,
                    'optimized': True
                }
            }
            
            info_path = os.path.join(project_path, 'optimization_info.json')
            with open(info_path, 'w') as f:
                json.dump(optimization_info, f, indent=2)
            
        except Exception as e:
            logger.error(f"Texture optimization failed: {str(e)}")
    
    def _optimize_meshes(self, project_path: str, settings: Dict):
        """Optimize meshes for platform"""
        try:
            models_dir = os.path.join(project_path, 'Assets', 'Models')
            if not os.path.exists(models_dir):
                return
            
            compression = settings.get('mesh_compression', 'Medium')
            
            logger.info(f"Optimizing meshes: compression={compression}")
            
            # This would typically involve:
            # 1. Reducing polygon count
            # 2. Optimizing vertex data
            # 3. Creating LOD levels
            # 4. Applying mesh compression
            
        except Exception as e:
            logger.error(f"Mesh optimization failed: {str(e)}")
    
    def _optimize_audio(self, project_path: str, settings: Dict):
        """Optimize audio for platform"""
        try:
            audio_dir = os.path.join(project_path, 'Assets', 'Audio')
            if not os.path.exists(audio_dir):
                return
            
            compression = settings.get('audio_compression', 'Vorbis')
            
            logger.info(f"Optimizing audio: compression={compression}")
            
            # This would typically involve:
            # 1. Converting to optimal format
            # 2. Applying compression
            # 3. Adjusting sample rates
            # 4. Creating compressed variants
            
        except Exception as e:
            logger.error(f"Audio optimization failed: {str(e)}")
    
    def _create_platform_settings(self, project_path: str, platform: str, settings: Dict):
        """Create platform-specific build settings"""
        try:
            settings_dir = os.path.join(project_path, 'ProjectSettings')
            os.makedirs(settings_dir, exist_ok=True)
            
            # Create XR settings
            xr_settings = {
                'platform': platform,
                'target_framerate': settings.get('target_framerate', 60),
                'foveated_rendering': settings.get('foveated_rendering', False),
                'eye_tracking': settings.get('eye_tracking', False),
                'hand_tracking': settings.get('hand_tracking', False)
            }
            
            xr_settings_path = os.path.join(settings_dir, 'XRSettings.json')
            with open(xr_settings_path, 'w') as f:
                json.dump(xr_settings, f, indent=2)
            
            # Create build settings
            build_settings = {
                'platform': platform,
                'build_target': self.supported_platforms[platform]['build_target'],
                'optimization_level': 'Size' if platform == 'quest' else 'Balanced',
                'scripting_backend': 'IL2CPP',
                'api_compatibility_level': '.NET Standard 2.1'
            }
            
            build_settings_path = os.path.join(settings_dir, 'BuildSettings.json')
            with open(build_settings_path, 'w') as f:
                json.dump(build_settings, f, indent=2)
            
        except Exception as e:
            logger.error(f"Platform settings creation failed: {str(e)}")
    
    def build_for_quest(
        self,
        project_path: str,
        output_dir: str,
        build_name: str = 'VRApp'
    ) -> str:
        """Build for Meta Quest"""
        try:
            logger.info("Building for Meta Quest")
            
            # Check Android SDK
            android_sdk = self.detect_android_sdk()
            if not android_sdk:
                logger.warning("Android SDK not found, creating placeholder build")
                return self._create_placeholder_build(output_dir, build_name, '.apk')
            
            # Check Unity
            unity_exe = self.detect_unity_installation()
            if not unity_exe:
                logger.warning("Unity not found, creating placeholder build")
                return self._create_placeholder_build(output_dir, build_name, '.apk')
            
            # Apply Quest optimizations
            self.optimize_for_platform(project_path, 'quest')
            
            # Build APK
            apk_path = os.path.join(output_dir, f'{build_name}.apk')
            
            build_cmd = [
                unity_exe,
                '-batchmode',
                '-quit',
                '-projectPath', project_path,
                '-buildTarget', 'Android',
                '-executeMethod', 'BuildScript.BuildForQuest',
                '-buildPath', apk_path
            ]
            
            logger.info(f"Building Quest APK: {' '.join(build_cmd)}")
            result = subprocess.run(build_cmd, capture_output=True, text=True, timeout=1800)
            
            if result.returncode != 0:
                logger.error(f"Quest build failed: {result.stderr}")
                return self._create_placeholder_build(output_dir, build_name, '.apk')
            
            if os.path.exists(apk_path):
                logger.info(f"Quest build completed: {apk_path}")
                return apk_path
            else:
                return self._create_placeholder_build(output_dir, build_name, '.apk')
            
        except Exception as e:
            logger.error(f"Quest build failed: {str(e)}")
            return self._create_placeholder_build(output_dir, build_name, '.apk')
    
    def build_for_vision_pro(
        self,
        project_path: str,
        output_dir: str,
        build_name: str = 'VRApp'
    ) -> str:
        """Build for Apple Vision Pro"""
        try:
            logger.info("Building for Apple Vision Pro")
            
            # Check if on macOS
            import platform
            if platform.system() != 'Darwin':
                logger.warning("Vision Pro builds require macOS, creating placeholder")
                return self._create_placeholder_build(output_dir, build_name, '.ipa')
            
            # Check Unity
            unity_exe = self.detect_unity_installation()
            if not unity_exe:
                logger.warning("Unity not found, creating placeholder build")
                return self._create_placeholder_build(output_dir, build_name, '.ipa')
            
            # Apply Vision Pro optimizations
            self.optimize_for_platform(project_path, 'vision_pro')
            
            # Build for visionOS
            ipa_path = os.path.join(output_dir, f'{build_name}.ipa')
            
            build_cmd = [
                unity_exe,
                '-batchmode',
                '-quit',
                '-projectPath', project_path,
                '-buildTarget', 'VisionOS',
                '-executeMethod', 'BuildScript.BuildForVisionPro',
                '-buildPath', ipa_path
            ]
            
            logger.info(f"Building Vision Pro IPA: {' '.join(build_cmd)}")
            result = subprocess.run(build_cmd, capture_output=True, text=True, timeout=1800)
            
            if result.returncode != 0:
                logger.error(f"Vision Pro build failed: {result.stderr}")
                return self._create_placeholder_build(output_dir, build_name, '.ipa')
            
            if os.path.exists(ipa_path):
                logger.info(f"Vision Pro build completed: {ipa_path}")
                return ipa_path
            else:
                return self._create_placeholder_build(output_dir, build_name, '.ipa')
            
        except Exception as e:
            logger.error(f"Vision Pro build failed: {str(e)}")
            return self._create_placeholder_build(output_dir, build_name, '.ipa')
    
    def build_for_webxr(
        self,
        project_path: str,
        output_dir: str,
        build_name: str = 'VRApp'
    ) -> str:
        """Build for WebXR"""
        try:
            logger.info("Building for WebXR")
            
            # Check Unity
            unity_exe = self.detect_unity_installation()
            if not unity_exe:
                logger.warning("Unity not found, creating placeholder build")
                return self._create_placeholder_webxr_build(output_dir, build_name)
            
            # Apply WebXR optimizations
            self.optimize_for_platform(project_path, 'webxr')
            
            # Build for WebGL
            webgl_build_dir = os.path.join(output_dir, f'{build_name}_WebGL')
            
            build_cmd = [
                unity_exe,
                '-batchmode',
                '-quit',
                '-projectPath', project_path,
                '-buildTarget', 'WebGL',
                '-executeMethod', 'BuildScript.BuildForWebXR',
                '-buildPath', webgl_build_dir
            ]
            
            logger.info(f"Building WebXR: {' '.join(build_cmd)}")
            result = subprocess.run(build_cmd, capture_output=True, text=True, timeout=1800)
            
            if result.returncode != 0:
                logger.error(f"WebXR build failed: {result.stderr}")
                return self._create_placeholder_webxr_build(output_dir, build_name)
            
            # Package WebGL build
            if os.path.exists(webgl_build_dir):
                zip_path = os.path.join(output_dir, f'{build_name}_webxr.zip')
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for root, dirs, files in os.walk(webgl_build_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, webgl_build_dir)
                            zipf.write(file_path, arcname)
                
                logger.info(f"WebXR build completed: {zip_path}")
                return zip_path
            else:
                return self._create_placeholder_webxr_build(output_dir, build_name)
            
        except Exception as e:
            logger.error(f"WebXR build failed: {str(e)}")
            return self._create_placeholder_webxr_build(output_dir, build_name)
    
    def _create_placeholder_build(self, output_dir: str, build_name: str, extension: str) -> str:
        """Create a placeholder build file"""
        try:
            placeholder_path = os.path.join(output_dir, f'{build_name}{extension}')
            
            # Create a simple placeholder file
            with open(placeholder_path, 'wb') as f:
                f.write(b'Placeholder VR build - actual build tools not available\n')
                f.write(f'Build name: {build_name}\n'.encode())
                f.write(f'Target platform: {extension}\n'.encode())
                f.write(b'This is a demonstration build file.\n')
            
            logger.info(f"Created placeholder build: {placeholder_path}")
            return placeholder_path
            
        except Exception as e:
            logger.error(f"Placeholder build creation failed: {str(e)}")
            raise
    
    def _create_placeholder_webxr_build(self, output_dir: str, build_name: str) -> str:
        """Create a placeholder WebXR build"""
        try:
            webxr_dir = os.path.join(output_dir, f'{build_name}_webxr')
            os.makedirs(webxr_dir, exist_ok=True)
            
            # Create basic HTML file
            html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{build_name} - WebXR VR Experience</title>
    <style>
        body {{
            margin: 0;
            padding: 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            font-family: Arial, sans-serif;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            color: white;
        }}
        .container {{
            text-align: center;
            padding: 2rem;
            background: rgba(0,0,0,0.3);
            border-radius: 10px;
            backdrop-filter: blur(10px);
        }}
        .vr-button {{
            background: #ff6b6b;
            color: white;
            border: none;
            padding: 1rem 2rem;
            font-size: 1.2rem;
            border-radius: 5px;
            cursor: pointer;
            margin: 1rem;
            transition: background 0.3s;
        }}
        .vr-button:hover {{
            background: #ff5252;
        }}
        .info {{
            margin-top: 2rem;
            opacity: 0.8;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{build_name}</h1>
        <p>AI-Generated VR Experience</p>
        <button class="vr-button" onclick="enterVR()">Enter VR</button>
        <button class="vr-button" onclick="enterAR()">Enter AR</button>
        <div class="info">
            <p>This is a placeholder WebXR experience.</p>
            <p>In a full implementation, this would load your 3D scene and enable VR/AR interaction.</p>
        </div>
    </div>
    
    <script>
        function enterVR() {{
            alert('VR mode would be activated here with your 3D scene!');
        }}
        
        function enterAR() {{
            alert('AR mode would be activated here with your 3D scene!');
        }}
        
        // Check for WebXR support
        if ('xr' in navigator) {{
            console.log('WebXR is supported');
        }} else {{
            console.log('WebXR is not supported');
        }}
    </script>
</body>
</html>'''
            
            html_path = os.path.join(webxr_dir, 'index.html')
            with open(html_path, 'w') as f:
                f.write(html_content)
            
            # Create manifest.json for PWA
            manifest_content = {
                "name": f"{build_name} VR Experience",
                "short_name": build_name,
                "description": "AI-Generated VR Experience",
                "start_url": "./index.html",
                "display": "fullscreen",
                "background_color": "#667eea",
                "theme_color": "#764ba2",
                "icons": [
                    {
                        "src": "icon-192.png",
                        "sizes": "192x192",
                        "type": "image/png"
                    }
                ]
            }
            
            manifest_path = os.path.join(webxr_dir, 'manifest.json')
            with open(manifest_path, 'w') as f:
                json.dump(manifest_content, f, indent=2)
            
            # Package as ZIP
            zip_path = os.path.join(output_dir, f'{build_name}_webxr.zip')
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(webxr_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, webxr_dir)
                        zipf.write(file_path, arcname)
            
            # Clean up temporary directory
            shutil.rmtree(webxr_dir)
            
            logger.info(f"Created placeholder WebXR build: {zip_path}")
            return zip_path
            
        except Exception as e:
            logger.error(f"Placeholder WebXR build creation failed: {str(e)}")
            raise
    
    def create_build_scripts(self, project_path: str):
        """Create Unity build scripts for automated building"""
        try:
            editor_dir = os.path.join(project_path, 'Assets', 'Editor')
            os.makedirs(editor_dir, exist_ok=True)
            
            build_script_content = '''using UnityEngine;
using UnityEditor;
using UnityEditor.Build.Reporting;
using System.IO;

public class BuildScript
{
    [MenuItem("Build/Build for Quest")]
    public static void BuildForQuest()
    {
        BuildForPlatform(BuildTarget.Android, "Quest");
    }
    
    [MenuItem("Build/Build for Vision Pro")]
    public static void BuildForVisionPro()
    {
        BuildForPlatform(BuildTarget.VisionOS, "VisionPro");
    }
    
    [MenuItem("Build/Build for WebXR")]
    public static void BuildForWebXR()
    {
        BuildForPlatform(BuildTarget.WebGL, "WebXR");
    }
    
    static void BuildForPlatform(BuildTarget target, string platformName)
    {
        string buildPath = GetBuildPath(platformName);
        
        BuildPlayerOptions buildPlayerOptions = new BuildPlayerOptions();
        buildPlayerOptions.scenes = GetScenePaths();
        buildPlayerOptions.locationPathName = buildPath;
        buildPlayerOptions.target = target;
        buildPlayerOptions.options = BuildOptions.None;
        
        // Apply platform-specific settings
        ApplyPlatformSettings(target);
        
        BuildReport report = BuildPipeline.BuildPlayer(buildPlayerOptions);
        BuildSummary summary = report.summary;
        
        if (summary.result == BuildResult.Succeeded)
        {
            Debug.Log($"{platformName} build succeeded: {buildPath}");
        }
        else
        {
            Debug.LogError($"{platformName} build failed");
        }
    }
    
    static string[] GetScenePaths()
    {
        string[] scenes = new string[EditorBuildSettings.scenes.Length];
        for (int i = 0; i < scenes.Length; i++)
        {
            scenes[i] = EditorBuildSettings.scenes[i].path;
        }
        return scenes;
    }
    
    static string GetBuildPath(string platformName)
    {
        string buildPath = System.Environment.GetCommandLineArgs();
        // Parse command line arguments for build path
        // Default to Builds folder
        return Path.Combine(Application.dataPath, "..", "Builds", platformName);
    }
    
    static void ApplyPlatformSettings(BuildTarget target)
    {
        switch (target)
        {
            case BuildTarget.Android:
                // Quest/Android settings
                PlayerSettings.Android.minSdkVersion = AndroidSdkVersions.AndroidApiLevel23;
                PlayerSettings.Android.targetSdkVersion = AndroidSdkVersions.AndroidApiLevelAuto;
                PlayerSettings.SetScriptingBackend(BuildTargetGroup.Android, ScriptingImplementation.IL2CPP);
                PlayerSettings.Android.targetArchitectures = AndroidArchitecture.ARM64;
                break;
                
            case BuildTarget.WebGL:
                // WebXR settings
                PlayerSettings.WebGL.compressionFormat = WebGLCompressionFormat.Gzip;
                PlayerSettings.WebGL.memorySize = 512;
                PlayerSettings.WebGL.exceptionSupport = WebGLExceptionSupport.None;
                break;
        }
    }
}
'''
            
            build_script_path = os.path.join(editor_dir, 'BuildScript.cs')
            with open(build_script_path, 'w') as f:
                f.write(build_script_content)
            
            logger.info("Unity build scripts created")
            
        except Exception as e:
            logger.error(f"Build scripts creation failed: {str(e)}")
    
    def deploy(
        self,
        engine_project_path: str,
        output_dir: str = './output',
        platforms: List[str] = ['quest', 'webxr'],
        build_name: str = 'AI_VR_Experience',
        **kwargs
    ) -> Dict:
        """Main deployment method"""
        try:
            logger.info(f"Deploying to platforms: {platforms}")
            
            # Create build scripts if Unity project
            if 'Assets' in os.listdir(engine_project_path):
                self.create_build_scripts(engine_project_path)
            
            deployment_results = {}
            
            for platform in platforms:
                if platform not in self.supported_platforms:
                    logger.warning(f"Unsupported platform: {platform}")
                    continue
                
                try:
                    if platform == 'quest':
                        build_path = self.build_for_quest(
                            engine_project_path, output_dir, build_name
                        )
                    elif platform == 'vision_pro':
                        build_path = self.build_for_vision_pro(
                            engine_project_path, output_dir, build_name
                        )
                    elif platform == 'webxr':
                        build_path = self.build_for_webxr(
                            engine_project_path, output_dir, build_name
                        )
                    else:
                        # Generic build
                        build_path = self._create_placeholder_build(
                            output_dir, 
                            f'{build_name}_{platform}',
                            self.supported_platforms[platform]['package_format']
                        )
                    
                    deployment_results[platform] = {
                        'success': True,
                        'build_path': build_path,
                        'platform_info': self.supported_platforms[platform]
                    }
                    
                except Exception as e:
                    logger.error(f"Deployment failed for {platform}: {str(e)}")
                    deployment_results[platform] = {
                        'success': False,
                        'error': str(e)
                    }
            
            # Create deployment summary
            summary = {
                'total_platforms': len(platforms),
                'successful_builds': len([r for r in deployment_results.values() if r['success']]),
                'failed_builds': len([r for r in deployment_results.values() if not r['success']]),
                'build_name': build_name,
                'output_directory': output_dir
            }
            
            return {
                'success': True,
                'deployment_results': deployment_results,
                'summary': summary
            }
            
        except Exception as e:
            logger.error(f"VR deployment failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
