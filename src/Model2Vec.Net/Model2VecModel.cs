using System.Numerics.Tensors;
using System.Text.Json;

namespace Model2VecNet;

/// <summary>
/// Loads and runs Model2Vec static embedding models.
/// </summary>
public sealed partial class Model2VecModel
{
    private readonly float[] _embeddings;
    private readonly float[]? _weights;
    private readonly int[]? _mapping;
    private readonly Model2VecTokenizer _tokenizer;
    private readonly bool _normalize;

    private Model2VecModel(float[] embeddings, int rowCount, int dimension, float[]? weights, int[]? mapping, Model2VecTokenizer tokenizer, bool normalize)
    {
        _embeddings = embeddings;
        RowCount = rowCount;
        Dimension = dimension;
        _weights = weights;
        _mapping = mapping;
        _tokenizer = tokenizer;
        _normalize = normalize;
    }

    /// <summary>
    /// Gets the number of dimensions in each output embedding.
    /// </summary>
    public int Dimension { get; }

    /// <summary>
    /// Gets the number of embedding rows loaded from the model.
    /// </summary>
    public int RowCount { get; }

    /// <summary>
    /// Loads a Model2Vec model from a folder, or from a <c>model.safetensors</c> file path within a model folder.
    /// </summary>
    /// <param name="directoryOrPath">The model folder or safetensors file path.</param>
    /// <returns>The loaded model.</returns>
    public static Model2VecModel Load(string directoryOrPath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(directoryOrPath);

        string modelFile;
        string directory;
        if (File.Exists(directoryOrPath))
        {
            modelFile = directoryOrPath;
            directory = Path.GetDirectoryName(Path.GetFullPath(directoryOrPath)) ?? Directory.GetCurrentDirectory();
        }
        else
        {
            directory = directoryOrPath;
            modelFile = Path.Combine(directory, "model.safetensors");
        }

        string tokenizerFile = Path.Combine(directory, "tokenizer.json");
        string configFile = Path.Combine(directory, "config.json");
        if (!File.Exists(modelFile))
        {
            throw new FileNotFoundException("Could not find model.safetensors.", modelFile);
        }
        if (!File.Exists(tokenizerFile))
        {
            throw new FileNotFoundException("Could not find tokenizer.json.", tokenizerFile);
        }
        if (!File.Exists(configFile))
        {
            throw new FileNotFoundException("Could not find config.json.", configFile);
        }

        SafetensorsFile tensors = SafetensorsFile.Load(modelFile);
        SafetensorTensor embeddingTensor = tensors.GetTensorOrDefault("embeddings") ?? tensors.GetTensor("embedding.weight");
        if (embeddingTensor.Shape.Length != 2)
        {
            throw new InvalidDataException("The embedding tensor must be two-dimensional.");
        }

        int rowCount = checked((int)embeddingTensor.Shape[0]);
        int dimension = checked((int)embeddingTensor.Shape[1]);
        float[] embeddings = embeddingTensor.ToSingleArray();

        float[]? weights = tensors.GetTensorOrDefault("weights")?.ToSingleArray();
        int[]? mapping = tensors.GetTensorOrDefault("mapping")?.ToInt32Array();
        Model2VecTokenizer tokenizer = Model2VecTokenizer.Load(tokenizerFile);
        bool normalize = ReadNormalize(configFile);

        if (mapping is null && rowCount != tokenizer.VocabularyCount)
        {
            throw new InvalidDataException($"The embedding row count ({rowCount}) does not match the tokenizer vocabulary count ({tokenizer.VocabularyCount}).");
        }
        if (weights is not null && weights.Length != tokenizer.VocabularyCount)
        {
            throw new InvalidDataException($"The weights tensor length ({weights.Length}) does not match the tokenizer vocabulary count ({tokenizer.VocabularyCount}).");
        }
        if (mapping is not null && mapping.Length != tokenizer.VocabularyCount)
        {
            throw new InvalidDataException($"The mapping tensor length ({mapping.Length}) does not match the tokenizer vocabulary count ({tokenizer.VocabularyCount}).");
        }

        return new Model2VecModel(embeddings, rowCount, dimension, weights, mapping, tokenizer, normalize);
    }

    /// <summary>
    /// Encodes text as a static embedding.
    /// </summary>
    /// <param name="text">The text to encode.</param>
    /// <param name="maxLength">The maximum number of tokenizer tokens to pool. The Model2Vec default is 512.</param>
    /// <returns>The pooled embedding.</returns>
    public float[] Encode(string text, int? maxLength = 512)
    {
        ArgumentNullException.ThrowIfNull(text);
        int[] ids = _tokenizer.Tokenize(text, maxLength);
        return EncodeTokenIds(ids);
    }

    /// <summary>
    /// Encodes a batch of texts as static embeddings.
    /// </summary>
    /// <param name="texts">The texts to encode.</param>
    /// <param name="maxLength">The maximum number of tokenizer tokens to pool. The Model2Vec default is 512.</param>
    /// <returns>One pooled embedding per input text.</returns>
    public float[][] Encode(IReadOnlyList<string> texts, int? maxLength = 512)
    {
        ArgumentNullException.ThrowIfNull(texts);
        var result = new float[texts.Count][];
        for (int i = 0; i < texts.Count; i++)
        {
            result[i] = Encode(texts[i], maxLength);
        }

        return result;
    }

    private float[] EncodeTokenIds(int[] ids)
    {
        var output = new float[Dimension];
        if (ids.Length == 0)
        {
            return output;
        }

        foreach (int tokenId in ids)
        {
            int row = _mapping is null ? tokenId : _mapping[tokenId];
            if ((uint)row >= (uint)RowCount)
            {
                throw new InvalidDataException($"Token id {tokenId} maps to embedding row {row}, outside the embedding matrix.");
            }

            float weight = _weights is null ? 1.0f : _weights[tokenId];
            int rowOffset = row * Dimension;
            for (int d = 0; d < Dimension; d++)
            {
                output[d] += _embeddings[rowOffset + d] * weight;
            }
        }

        TensorPrimitives.Multiply(output, 1.0f / ids.Length, output);

        if (_normalize)
        {
            float norm = TensorPrimitives.Norm(output);
            if (norm > 0)
            {
                TensorPrimitives.Divide(output, norm, output);
            }
        }

        return output;
    }

    private static bool ReadNormalize(string configFile)
    {
        using FileStream stream = File.OpenRead(configFile);
        using JsonDocument document = JsonDocument.Parse(stream);
        return document.RootElement.TryGetProperty("normalize", out JsonElement normalize) &&
               normalize.ValueKind == JsonValueKind.True;
    }
}
