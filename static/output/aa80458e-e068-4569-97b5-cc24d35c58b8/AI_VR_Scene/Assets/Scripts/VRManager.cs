using UnityEngine;
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
