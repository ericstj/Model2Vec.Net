using System.Text.Json;
using Model2VecNet;
using Xunit;

namespace Model2Vec.Net.Tests;

public sealed class Model2VecOracleTests
{
    private const double Tolerance = 1e-4;

    public static TheoryData<ModelCase> Models => new()
    {
        new ModelCase("potion-base-2M", "oracle_potion_base_2m.json", "MODEL2VEC_POTION_BASE_2M"),
        new ModelCase("distilroberta-base-ca-v2", "oracle_distilroberta_base_ca_v2.json", "MODEL2VEC_DISTILROBERTA_BASE_CA_V2")
    };

    [Theory]
    [MemberData(nameof(Models))]
    public void ModelLoads(ModelCase modelCase)
    {
        Model2VecModel model = Model2VecModel.Load(ModelPath(modelCase));
        OracleData oracle = LoadOracle(modelCase);

        Assert.Equal(oracle.dimension, model.Dimension);
    }

    [Theory]
    [MemberData(nameof(Models))]
    public void EmbeddingsMatchPythonOracle(ModelCase modelCase)
    {
        Model2VecModel model = Model2VecModel.Load(ModelPath(modelCase));
        OracleData oracle = LoadOracle(modelCase);

        foreach (OracleCase testCase in oracle.cases)
        {
            float[] actual = model.Encode(testCase.text);
            Assert.Equal(oracle.dimension, actual.Length);
            Assert.Equal(testCase.embedding.Count, actual.Length);

            for (int i = 0; i < actual.Length; i++)
            {
                double diff = Math.Abs(testCase.embedding[i] - actual[i]);
                Assert.True(diff < Tolerance, $"[{testCase.text}] dim {i}: expected {testCase.embedding[i]}, got {actual[i]}, diff {diff}");
            }
        }
    }

    [Theory]
    [MemberData(nameof(Models))]
    public void EmptyStringEncodesAsZeroVector(ModelCase modelCase)
    {
        Model2VecModel model = Model2VecModel.Load(ModelPath(modelCase));
        float[] actual = model.Encode("");

        Assert.Equal(model.Dimension, actual.Length);
        Assert.All(actual, value => Assert.Equal(0.0f, value));
    }

    [Fact]
    public void TokenizerIdsMatchReferenceSamples()
    {
        var potion = new ModelCase("potion-base-2M", "oracle_potion_base_2m.json", "MODEL2VEC_POTION_BASE_2M");
        var distilRoberta = new ModelCase("distilroberta-base-ca-v2", "oracle_distilroberta_base_ca_v2.json", "MODEL2VEC_DISTILROBERTA_BASE_CA_V2");

        Assert.Equal(
            [6674, 1145, 2906, 23734],
            Tokenize(ModelPath(potion), "café déjà vu"));
        Assert.Equal(
            [597, 10528, 38804],
            Tokenize(ModelPath(distilRoberta), "Hello world"));
        Assert.Equal(
            [4339, 976, 20283, 52],
            Tokenize(ModelPath(distilRoberta), " Olá mundo!"));
    }

    private static string ModelPath(ModelCase modelCase)
    {
        string baseDir = AppContext.BaseDirectory;
        string local = Path.Combine(baseDir, "models", modelCase.Name);
        if (File.Exists(Path.Combine(local, "model.safetensors")))
        {
            return local;
        }

        string? env = Environment.GetEnvironmentVariable(modelCase.EnvironmentVariable);
        if (!string.IsNullOrEmpty(env) && File.Exists(Path.Combine(env, "model.safetensors")))
        {
            return env;
        }

        throw new SkipException($"{modelCase.Name} model files not found. Build should download them, or set {modelCase.EnvironmentVariable}.");
    }

    private static OracleData LoadOracle(ModelCase modelCase)
    {
        string path = Path.Combine(AppContext.BaseDirectory, modelCase.OracleFile);
        using FileStream stream = File.OpenRead(path);
        return JsonSerializer.Deserialize<OracleData>(stream)!;
    }

    private static int[] Tokenize(string modelPath, string text)
    {
        Type tokenizerType = typeof(Model2VecModel).Assembly.GetType("Model2VecNet.Model2VecTokenizer")!;
        object tokenizer = tokenizerType.GetMethod("Load")!.Invoke(null, [Path.Combine(modelPath, "tokenizer.json")])!;
        return (int[])tokenizerType.GetMethod("Tokenize")!.Invoke(tokenizer, [text, 512])!;
    }
}

public sealed record ModelCase(string Name, string OracleFile, string EnvironmentVariable);

public sealed class OracleData
{
    public string model { get; set; } = "";
    public int dimension { get; set; }
    public bool normalize { get; set; }
    public List<OracleCase> cases { get; set; } = new();
}

public sealed class OracleCase
{
    public string text { get; set; } = "";
    public List<double> embedding { get; set; } = new();
}

[System.Diagnostics.CodeAnalysis.SuppressMessage("Design", "CA1032")]
public sealed class SkipException(string message) : Exception(message);
