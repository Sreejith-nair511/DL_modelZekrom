using UnityEngine;
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
