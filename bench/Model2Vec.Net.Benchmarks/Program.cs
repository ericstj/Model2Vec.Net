using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;
using Model2VecNet;

namespace Model2VecNet.Benchmarks;

[MemoryDiagnoser]
public class EncodeBenchmarks
{
    private Model2VecModel _model = null!;

    private const string ShortText = "Model2Vec turns token embeddings into fast sentence vectors.";
    private const string LongText =
        "Model2Vec is a static embedding technique that tokenizes text, pools token embeddings, " +
        "applies optional vocabulary quantization weights, and returns normalized vectors for retrieval, " +
        "classification, clustering, and other natural language workloads across diverse inputs.";

    private static readonly string[] Corpus =
    {
        "Hello, how are you doing today?",
        "Bonjour, comment allez-vous aujourd'hui?",
        "Hola, ¿cómo estás hoy?",
        "Olá mundo! Acentos: café, ação, coração.",
        "Guten Tag, wie geht es Ihnen heute?",
        "Привет, как у тебя дела сегодня?",
        "こんにちは、お元気ですか。",
        "你好，今天过得怎么样？",
        "Numbers 123 456, punctuation, and MIXED case!",
        "Model2Vec.Net loads safetensors and Hugging Face tokenizers.",
    };

    [GlobalSetup]
    public void Setup() => _model = Model2VecModel.Load(ModelLocator.Path());

    [Benchmark(Baseline = true)]
    public float[] EncodeSingleShortText() => _model.Encode(ShortText);

    [Benchmark]
    public float[] EncodeSingleLongText() => _model.Encode(LongText);

    [Benchmark]
    public float[][] EncodeBatch() => _model.Encode(Corpus);
}

public class LoadBenchmarks
{
    [Benchmark]
    public Model2VecModel LoadModel() => Model2VecModel.Load(ModelLocator.Path());
}

internal static class ModelLocator
{
    public static string Path()
    {
        string local = System.IO.Path.Combine(AppContext.BaseDirectory, "models", "potion-base-2M");
        if (File.Exists(System.IO.Path.Combine(local, "model.safetensors")))
        {
            return local;
        }

        string? env = Environment.GetEnvironmentVariable("MODEL2VEC_POTION_BASE_2M");
        if (!string.IsNullOrEmpty(env) && File.Exists(System.IO.Path.Combine(env, "model.safetensors")))
        {
            return env;
        }

        throw new FileNotFoundException("potion-base-2M not found. Place it under bench/.../models or set MODEL2VEC_POTION_BASE_2M.");
    }
}

public static class Program
{
    public static void Main(string[] args) =>
        BenchmarkSwitcher.FromAssembly(typeof(Program).Assembly).Run(args);
}
