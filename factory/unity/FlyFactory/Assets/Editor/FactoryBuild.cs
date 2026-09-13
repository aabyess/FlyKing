using System.IO;
using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEditor.SceneManagement;
using UnityEngine;

/// <summary>
/// 공장 씬 만들기·맥 빌드(배치 모드에서도).
///   Unity -batchmode -projectPath factory/unity/FlyFactory -executeMethod FactoryBuild.BuildMac -quit -logFile -
/// 빌드 실행: Build/FlyFactory.app --args -shot /경로/shot.png -shotAfter 40   (40초 뒤 화면을 찍고 종료)
/// </summary>
public static class FactoryBuild
{
    const string ScenePath = "Assets/Scenes/Factory.unity";

    [MenuItem("FlyFactory/공장 씬 만들기")]
    public static void CreateScene()
    {
        var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
        var game = new GameObject("초파리 공장");
        game.AddComponent<BrainClient>();
        game.AddComponent<FactoryGame>();
        Directory.CreateDirectory("Assets/Scenes");
        EditorSceneManager.SaveScene(scene, ScenePath);
        EditorBuildSettings.scenes = new[] { new EditorBuildSettingsScene(ScenePath, true) };
        AssetDatabase.SaveAssets();
        Debug.Log("SCENE_CREATED " + ScenePath);
    }

    [MenuItem("FlyFactory/맥 빌드")]
    public static void BuildMac()
    {
        CreateScene();
        PlayerSettings.productName = "초파리 공장";
        PlayerSettings.fullScreenMode = FullScreenMode.Windowed;
        PlayerSettings.defaultScreenWidth = 1600;
        PlayerSettings.defaultScreenHeight = 1000;
        PlayerSettings.resizableWindow = true;
        PlayerSettings.runInBackground = true;
        var report = BuildPipeline.BuildPlayer(new BuildPlayerOptions
        {
            scenes = new[] { ScenePath },
            locationPathName = "Build/FlyFactory.app",
            target = BuildTarget.StandaloneOSX,
            options = BuildOptions.None,
        });
        Debug.Log($"BUILD_RESULT {report.summary.result} errors={report.summary.totalErrors} size={report.summary.totalSize}");
        if (Application.isBatchMode) EditorApplication.Exit(report.summary.result == BuildResult.Succeeded ? 0 : 1);
    }
}

/// <summary>FBX 가져오기 규칙: 원래 숫자 그대로(단위 변환 없음). 초파리는 레거시 애니메이션(Walk_Tripod·Idle_Groom).</summary>
public class FlyFactoryModelImport : AssetPostprocessor
{
    void OnPreprocessModel()
    {
        var mi = (ModelImporter)assetImporter;
        mi.useFileScale = false;
        mi.globalScale = 1f;
        if (assetPath.Contains("초파리"))
        {
            mi.animationType = ModelImporterAnimationType.Legacy;
            mi.importAnimation = true;
        }
        else
        {
            mi.animationType = ModelImporterAnimationType.None;
            mi.importAnimation = false;
        }
    }
}
