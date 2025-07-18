import os
import json
import shutil
import logging
from typing import Dict, List
import zipfile

logger = logging.getLogger(__name__)

class GameEngineExporter:
    def __init__(self):
        self.supported_engines = {
            'unity': {
                'name': 'Unity',
                'versions': ['2022.3.0f1', '2023.1.0f1'],
                'project_template': 'unity_template',
                'supported_formats': ['.fbx', '.obj', '.glb', '.dae']
            },
            'unreal': {
                'name': 'Unreal Engine',
                'versions': ['5.1', '5.2', '5.3'],
                'project_template': 'unreal_template',
                'supported_formats': ['.fbx', '.obj', '.glb', '.usd']
            }
        }
    
    def get_available_engines(self) -> List[Dict]:
        return list(self.supported_engines.values())
    
    def create_unity_project(
        self,
        project_name: str,
        output_dir: str,
        template: str = '3D'
    ) -> str:
        try:
            project_path = os.path.join(output_dir, project_name)
            
            directories = [
                'Assets',
                'Assets/Scripts',
                'Assets/Materials',
                'Assets/Models',
                'Assets/Scenes',
                'Assets/Textures',
                'Library',
                'Logs',
                'Packages',
                'ProjectSettings',
                'UserSettings'
            ]
            
            for directory in directories:
                os.makedirs(os.path.join(project_path, directory), exist_ok=True)
            
            self._create_unity_project_files(project_path)
            
            logger.info(f"Unity project created: {project_path}")
            return project_path
            
        except Exception as e:
            logger.error(f"Unity project creation failed: {str(e)}")
            raise
    
    def _create_unity_project_files(self, project_path: str):
        try:
            # ProjectSettings/ProjectVersion.txt
            project_version_path = os.path.join(project_path, 'ProjectSettings', 'ProjectVersion.txt')
            with open(project_version_path, 'w') as f:
                f.write("m_EditorVersion: 2022.3.0f1\n")
                f.write("m_EditorVersionWithRevision: 2022.3.0f1 (fb119bb0b476)\n")
            
            # Packages/manifest.json
            manifest = {
                "dependencies": {
                    "com.unity.render-pipelines.universal": "14.0.8",
                    "com.unity.xr.interaction.toolkit": "2.4.0",
                    "com.unity.xr.openxr": "1.8.2",
                    "com.unity.modules.ai": "1.0.0",
                    "com.unity.modules.animation": "1.0.0",
                    "com.unity.modules.audio": "1.0.0",
                    "com.unity.modules.physics": "1.0.0",
                    "com.unity.modules.ui": "1.0.0",
                    "com.unity.modules.vr": "1.0.0",
                    "com.unity.modules.xr": "1.0.0"
                }
            }
            
            manifest_path = os.path.join(project_path, 'Packages', 'manifest.json')
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
            
            self._create_unity_scene(project_path)
            self._create_unity_vr_scripts(project_path)
            
        except Exception as e:
            logger.error(f"Unity project files creation failed: {str(e)}")
            raise
    
    def _create_unity_scene(self, project_path: str):
        scene_content = '''%YAML 1.1
%TAG !u! tag:unity3d.com,2011:
--- !u!29 &1
OcclusionCullingSettings:
  m_ObjectHideFlags: 0
  serializedVersion: 2
--- !u!104 &2
RenderSettings:
  m_ObjectHideFlags: 0
  serializedVersion: 9
  m_Fog: 0
  m_FogColor: {r: 0.5, g: 0.5, b: 0.5, a: 1}
--- !u!1 &519420028
GameObject:
  m_ObjectHideFlags: 0
  m_Component:
  - component: {fileID: 519420032}
  - component: {fileID: 519420031}
  - component: {fileID: 519420029}
  m_Layer: 0
  m_Name: Main Camera
  m_TagString: MainCamera
--- !u!81 &519420029
AudioListener:
  m_ObjectHideFlags: 0
  m_GameObject: {fileID: 519420028}
  m_Enabled: 1
--- !u!20 &519420031
Camera:
  m_ObjectHideFlags: 0
  m_GameObject: {fileID: 519420028}
  m_Enabled: 1
  serializedVersion: 2
  m_ClearFlags: 1
  m_BackGroundColor: {r: 0.19215687, g: 0.3019608, b: 0.4745098, a: 0}
--- !u!4 &519420032
Transform:
  m_ObjectHideFlags: 0
  m_GameObject: {fileID: 519420028}
  m_LocalRotation: {x: 0, y: 0, z: 0, w: 1}
  m_LocalPosition: {x: 0, y: 1, z: -10}
  m_LocalScale: {x: 1, y: 1, z: 1}
'''
        
        scene_path = os.path.join(project_path, 'Assets', 'Scenes', 'MainScene.unity')
        with open(scene_path, 'w') as f:
            f.write(scene_content)
    
    def _create_unity_vr_scripts(self, project_path: str):
        try:
            scripts_dir = os.path.join(project_path, 'Assets', 'Scripts')
            
            vr_manager_script = '''using UnityEngine;
using UnityEngine.XR;

public class VRManager : MonoBehaviour
{
    [Header("VR Settings")]
    public bool enableVROnStart = true;
    
    [Header("Model Settings")]
    public GameObject modelPrefab;
    public Transform modelSpawnPoint;
    
    void Start()
    {
        if (enableVROnStart)
        {
            InitializeVR();
        }
        
        if (modelPrefab != null && modelSpawnPoint != null)
        {
            SpawnModel();
        }
    }
    
    void InitializeVR()
    {
        XRSettings.enabled = true;
    }
    
    void SpawnModel()
    {
        GameObject spawnedModel = Instantiate(modelPrefab, modelSpawnPoint.position, modelSpawnPoint.rotation);
        
        if (spawnedModel.GetComponent<Collider>() == null)
        {
            spawnedModel.AddComponent<MeshCollider>();
        }
    }
    
    public void LoadModel(string modelPath)
    {
        Debug.Log($"Loading model from: {modelPath}");
    }
}
'''
            
            vr_manager_path = os.path.join(scripts_dir, 'VRManager.cs')
            with open(vr_manager_path, 'w') as f:
                f.write(vr_manager_script)
            
            logger.info("Unity VR scripts created successfully")
            
        except Exception as e:
            logger.error(f"Unity VR scripts creation failed: {str(e)}")
            raise
    
    def import_model_to_unity(
        self,
        model_path: str,
        unity_project_path: str,
        target_folder: str = 'Models'
    ) -> str:
        try:
            assets_path = os.path.join(unity_project_path, 'Assets', target_folder)
            os.makedirs(assets_path, exist_ok=True)
            
            model_filename = os.path.basename(model_path)
            unity_model_path = os.path.join(assets_path, model_filename)
            shutil.copy2(model_path, unity_model_path)
            
            # Create meta file for Unity
            meta_content = f'''fileFormatVersion: 2
guid: {self._generate_unity_guid()}
ModelImporter:
  serializedVersion: 21300
  internalIDToNameTable: []
  externalObjects: {{}}
  materials:
    materialImportMode: 1
    materialName: 0
    materialSearch: 1
    materialLocation: 1
  meshes:
    lODScreenPercentages: []
    globalScale: 1
    meshCompression: 0
    addColliders: 0
    useSRGBMaterialColor: 1
    importVisibility: 1
    importBlendShapes: 1
    importCameras: 1
    importLights: 1
    fileIdsGeneration: 2
    swapUVChannels: 0
    generateSecondaryUV: 0
    useFileUnits: 1
    keepQuads: 0
    weldVertices: 1
    preserveHierarchy: 0
    skinWeightsMode: 0
    maxBonesPerVertex: 4
    minBoneWeight: 0.001
    meshOptimizationFlags: -1
    indexFormat: 0
  userData: 
  assetBundleName: 
  assetBundleVariant: 
'''
            
            meta_path = unity_model_path + '.meta'
            with open(meta_path, 'w') as f:
                f.write(meta_content)
            
            logger.info(f"Model imported to Unity: {unity_model_path}")
            return unity_model_path
            
        except Exception as e:
            logger.error(f"Unity model import failed: {str(e)}")
            raise
    
    def _generate_unity_guid(self) -> str:
        import uuid
        return str(uuid.uuid4()).replace('-', '')
    
    def package_unity_project(self, project_path: str, output_dir: str) -> str:
        try:
            project_name = os.path.basename(project_path)
            zip_path = os.path.join(output_dir, f'{project_name}_unity.zip')
            
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(project_path):
                    dirs[:] = [d for d in dirs if d not in ['Library', 'Temp', 'Logs']]
                    
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, project_path)
                        zipf.write(file_path, arcname)
            
            logger.info(f"Unity project packaged: {zip_path}")
            return zip_path
            
        except Exception as e:
            logger.error(f"Unity project packaging failed: {str(e)}")
            raise
    
    def export(
        self,
        model_path: str,
        output_dir: str = './output',
        engine: str = 'unity',
        project_name: str = 'AI_Generated_VR_Scene',
        **kwargs
    ) -> Dict:
        try:
            if engine not in self.supported_engines:
                raise ValueError(f"Unsupported engine: {engine}")
            
            logger.info(f"Exporting to {engine} engine")
            
            if engine == 'unity':
                project_path = self.create_unity_project(project_name, output_dir)
                
                imported_model_path = self.import_model_to_unity(
                    model_path, project_path, 'Models'
                )
                
                packaged_path = self.package_unity_project(project_path, output_dir)
                
                return {
                    'success': True,
                    'engine': engine,
                    'project_path': project_path,
                    'packaged_path': packaged_path,
                    'imported_model_path': imported_model_path
                }
                
            elif engine == 'unreal':
                project_path = os.path.join(output_dir, project_name)
                os.makedirs(project_path, exist_ok=True)
                models_dir = os.path.join(project_path, 'Models')
                os.makedirs(models_dir, exist_ok=True)

                # Copy the model file
                model_filename = os.path.basename(model_path)
                unreal_model_path = os.path.join(models_dir, model_filename)
                shutil.copy2(model_path, unreal_model_path)

                # Optionally, create a placeholder Unreal project file
                uproject_path = os.path.join(project_path, f'{project_name}.uproject')
                with open(uproject_path, 'w') as f:
                    f.write(json.dumps({
                        "FileVersion": 3,
                        "EngineAssociation": "5.2",
                        "Category": "VR",
                        "Description": "AI-generated VR scene"
                    }, indent=2))

                # Package the Unreal project as a zip
                zip_path = os.path.join(output_dir, f'{project_name}_unreal.zip')
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for root, dirs, files in os.walk(project_path):
                        for file in files:
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, project_path)
                            zipf.write(file_path, arcname)

                return {
                    'success': True,
                    'engine': engine,
                    'project_path': project_path,
                    'packaged_path': zip_path,
                    'imported_model_path': unreal_model_path
                }
        except Exception as e:
            logger.error(f"Export failed: {str(e)}")
            raise
