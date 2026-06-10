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
        new ModelCase("distilroberta-base-ca-v2", "oracle_distilroberta_base_ca_v2.json", "MODEL2VEC_DISTILROBERTA_BASE_CA_V2"),
        new ModelCase("potion-multilingual-128M", "oracle_potion_multilingual_128m.json", "MODEL2VEC_POTION_MULTILINGUAL_128M")
    };

    [SkippableTheory]
    [MemberData(nameof(Models))]
    public void ModelLoads(ModelCase modelCase)
    {
        Model2VecModel model = Model2VecModel.Load(ModelPath(modelCase));
        OracleData oracle = LoadOracle(modelCase);

        Assert.Equal(oracle.dimension, model.Dimension);
    }

    [SkippableTheory]
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

    [SkippableTheory]
    [MemberData(nameof(Models))]
    public void EmptyStringEncodesAsZeroVector(ModelCase modelCase)
    {
        Model2VecModel model = Model2VecModel.Load(ModelPath(modelCase));
        float[] actual = model.Encode("");

        Assert.Equal(model.Dimension, actual.Length);
        Assert.All(actual, value => Assert.Equal(0.0f, value));
    }

    [SkippableFact]
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

    [SkippableFact]
    public void UnigramTokenizerMatchesReference()
    {
        string dir = UnigramTokenizerDir();

        Assert.Equal([35378, 8999], Tokenize(dir, "Hello world"));
        Assert.Equal([91906, 3307, 38], Tokenize(dir, "Olá mundo!"));
        Assert.Equal([26216, 15154, 13946], Tokenize(dir, "café déjà vu"));
        Assert.Equal([6, 124084, 3221], Tokenize(dir, "你好世界"));
        Assert.Equal([103332, 7, 37638, 6, 121317, 38], Tokenize(dir, "Numbers 123 456!"));
        Assert.Equal([1813, 18454, 11373], Tokenize(dir, "Привет мир"));
    }

    [SkippableFact]
    public void JsonOnlyUnigramTokenizerMatchesReference()
    {
        var multilingual = new ModelCase("potion-multilingual-128M", "oracle_potion_multilingual_128m.json", "MODEL2VEC_POTION_MULTILINGUAL_128M");
        string dir = ModelPath(multilingual);

        Assert.Equal([35376, 8997], Tokenize(dir, "Hello world"));
        Assert.Equal([91904, 3305, 709], Tokenize(dir, "Olá mundo!"));
        Assert.Equal([26214, 15152, 13944], Tokenize(dir, "café déjà vu"));
        Assert.Equal([461602, 3219], Tokenize(dir, "你好世界"));
        Assert.Equal([103330, 5, 37636, 254776, 709], Tokenize(dir, "Numbers 123 456!"));
        Assert.Equal([1811, 18452, 11371], Tokenize(dir, "Привет мир"));

        // Added tokens are matched on the raw text before normalization; [UNK] (id 1) is then
        // stripped as an unknown-token id while [PAD] (id 0) is retained.
        Assert.Equal([250643, 0, 8997], Tokenize(dir, "hello [PAD] world"));
        Assert.Equal([8, 874], Tokenize(dir, "a [UNK] b"));
        Assert.Equal([1020, 111], Tokenize(dir, "x[UNK]y"));
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

        Skip.If(true, $"{modelCase.Name} model files not found. Build should download them, or set {modelCase.EnvironmentVariable}.");
        return local;
    }

    private static string UnigramTokenizerDir()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "models", "bge-m3-tokenizer");
        Skip.IfNot(
            File.Exists(Path.Combine(dir, "tokenizer.json")) && File.Exists(Path.Combine(dir, "sentencepiece.model")),
            "bge-m3 Unigram tokenizer fixture not found. Build should download tokenizer.json and sentencepiece.model.");
        return dir;
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
