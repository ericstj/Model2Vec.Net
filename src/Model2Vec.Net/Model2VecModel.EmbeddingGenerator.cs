using Microsoft.Extensions.AI;

namespace Model2VecNet;

/// <summary>
/// Implements <see cref="IEmbeddingGenerator{TInput, TEmbedding}"/> so a Model2Vec model can be
/// used directly with Microsoft.Extensions.AI and any consumer of that abstraction.
/// </summary>
public sealed partial class Model2VecModel : IEmbeddingGenerator<string, Embedding<float>>
{
    private EmbeddingGeneratorMetadata? _metadata;

    /// <inheritdoc />
    public Task<GeneratedEmbeddings<Embedding<float>>> GenerateAsync(
        IEnumerable<string> values,
        EmbeddingGenerationOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(values);
        cancellationToken.ThrowIfCancellationRequested();

        IReadOnlyList<string> inputs = values as IReadOnlyList<string> ?? values.ToList();
        float[][] vectors = Encode(inputs);

        var embeddings = new GeneratedEmbeddings<Embedding<float>>(vectors.Length);
        DateTimeOffset createdAt = DateTimeOffset.UtcNow;
        foreach (float[] vector in vectors)
        {
            embeddings.Add(new Embedding<float>(vector) { CreatedAt = createdAt });
        }

        return Task.FromResult(embeddings);
    }

    /// <inheritdoc />
    object? IEmbeddingGenerator.GetService(Type serviceType, object? serviceKey)
    {
        ArgumentNullException.ThrowIfNull(serviceType);

        return
            serviceKey is not null ? null :
            serviceType == typeof(EmbeddingGeneratorMetadata) ? _metadata ??= new EmbeddingGeneratorMetadata("Model2Vec", defaultModelDimensions: Dimension) :
            serviceType.IsInstanceOfType(this) ? this :
            null;
    }

    void IDisposable.Dispose()
    {
    }
}
