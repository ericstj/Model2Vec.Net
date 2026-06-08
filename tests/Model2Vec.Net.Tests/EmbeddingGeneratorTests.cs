using Microsoft.Extensions.AI;
using Model2VecNet;
using Xunit;

namespace Model2Vec.Net.Tests;

public sealed class EmbeddingGeneratorTests
{
    private static string PotionModelPath()
    {
        string local = Path.Combine(AppContext.BaseDirectory, "models", "potion-base-2M");
        if (File.Exists(Path.Combine(local, "model.safetensors")))
        {
            return local;
        }

        string? env = Environment.GetEnvironmentVariable("MODEL2VEC_POTION_BASE_2M");
        if (!string.IsNullOrEmpty(env) && File.Exists(Path.Combine(env, "model.safetensors")))
        {
            return env;
        }

        Skip.If(true, "potion-base-2M model files not found. Build should download them, or set MODEL2VEC_POTION_BASE_2M.");
        return local;
    }

    [SkippableFact]
    public async Task GeneratesEmbeddingMatchingEncode()
    {
        Model2VecModel model = Model2VecModel.Load(PotionModelPath());
        IEmbeddingGenerator<string, Embedding<float>> generator = model;

        const string text = "the quick brown fox";
        Embedding<float> embedding = await generator.GenerateAsync(text);
        float[] expected = model.Encode(text);

        Assert.Equal(model.Dimension, embedding.Vector.Length);
        Assert.Equal(expected, embedding.Vector.ToArray());
    }

    [SkippableFact]
    public async Task GeneratesBatchMatchingEncode()
    {
        Model2VecModel model = Model2VecModel.Load(PotionModelPath());
        IEmbeddingGenerator<string, Embedding<float>> generator = model;

        string[] inputs = ["hello world", "machine learning", "static embeddings"];
        GeneratedEmbeddings<Embedding<float>> embeddings = await generator.GenerateAsync(inputs);
        float[][] expected = model.Encode(inputs);

        Assert.Equal(inputs.Length, embeddings.Count);
        for (int i = 0; i < inputs.Length; i++)
        {
            Assert.Equal(expected[i], embeddings[i].Vector.ToArray());
        }
    }

    [SkippableFact]
    public void ExposesMetadata()
    {
        Model2VecModel model = Model2VecModel.Load(PotionModelPath());
        IEmbeddingGenerator generator = model;

        var metadata = generator.GetService(typeof(EmbeddingGeneratorMetadata)) as EmbeddingGeneratorMetadata;

        Assert.NotNull(metadata);
        Assert.Equal("Model2Vec", metadata!.ProviderName);
        Assert.Equal(model.Dimension, metadata.DefaultModelDimensions);
        Assert.Same(model, generator.GetService(typeof(Model2VecModel)));
    }
}
