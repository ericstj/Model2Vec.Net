using System.Text;
using System.Text.Json;
using Microsoft.ML.Tokenizers;

namespace Model2VecNet;

internal sealed class Model2VecTokenizer
{
    private static readonly char[] ByteLevelAlphabet = CreateByteLevelAlphabet();

    private readonly Tokenizer _tokenizer;
    private readonly HashSet<int> _unknownTokenIds;
    private readonly int _medianTokenLength;
    private readonly bool _byteLevelBpe;
    private readonly bool _addPrefixSpace;
    private readonly Dictionary<string, int>? _unigramVocab;
    private readonly TextNormalizer? _unigramNormalizer;
    private readonly IReadOnlyList<KeyValuePair<string, int>>? _unigramAddedTokens;

    private Model2VecTokenizer(
        Tokenizer tokenizer,
        int vocabularyCount,
        HashSet<int> unknownTokenIds,
        int medianTokenLength,
        bool byteLevelBpe = false,
        bool addPrefixSpace = false,
        Dictionary<string, int>? unigramVocab = null,
        TextNormalizer? unigramNormalizer = null,
        IReadOnlyDictionary<string, int>? unigramAddedTokens = null)
    {
        _tokenizer = tokenizer;
        VocabularyCount = vocabularyCount;
        _unknownTokenIds = unknownTokenIds;
        _medianTokenLength = medianTokenLength;
        _byteLevelBpe = byteLevelBpe;
        _addPrefixSpace = addPrefixSpace;
        _unigramVocab = unigramVocab;
        _unigramNormalizer = unigramNormalizer;

        // Match Hugging Face: special added tokens are matched on raw text before
        // normalization, longest content first so overlapping tokens prefer the longer match.
        _unigramAddedTokens = unigramAddedTokens is { Count: > 0 }
            ? unigramAddedTokens.OrderByDescending(static pair => pair.Key.Length).ToArray()
            : null;
    }

    public int VocabularyCount { get; }

    public static Model2VecTokenizer Load(string tokenizerJsonPath)
    {
        using FileStream stream = File.OpenRead(tokenizerJsonPath);
        using JsonDocument document = JsonDocument.Parse(stream);
        JsonElement root = document.RootElement;
        JsonElement model = root.GetProperty("model");
        Dictionary<string, int> vocab = ReadVocabulary(model.GetProperty("vocab"));
        HashSet<int> unknownTokenIds = ReadUnknownTokenIds(root, model, vocab);

        return model.GetProperty("type").GetString() switch
        {
            "WordPiece" => CreateWordPiece(root, model, vocab, unknownTokenIds),
            "BPE" => CreateBpe(root, model, vocab, unknownTokenIds),
            "Unigram" => CreateUnigram(tokenizerJsonPath, root, model, vocab, unknownTokenIds),
            string type => throw new NotSupportedException($"Tokenizer model type '{type}' is not supported."),
            null => throw new InvalidDataException("tokenizer.json model.type is missing.")
        };
    }

    public int[] Tokenize(string text, int? maxLength)
    {
        if (maxLength is not null)
        {
            int maxChars = checked(maxLength.GetValueOrDefault() * _medianTokenLength);
            if (text.Length > maxChars)
            {
                text = text[..maxChars];
            }
        }

        IReadOnlyList<int> encodedIds;
        if (_unigramVocab is not null)
        {
            encodedIds = EncodeUnigram(text);
        }
        else
        {
            encodedIds = _byteLevelBpe
                ? _tokenizer.EncodeToIds(ToByteLevelText(text, _addPrefixSpace), considerPreTokenization: false, considerNormalization: false)
                : _tokenizer.EncodeToIds(text, considerPreTokenization: true, considerNormalization: true);
        }

        var ids = new List<int>(encodedIds.Count);
        foreach (int id in encodedIds)
        {
            if (!_unknownTokenIds.Contains(id))
            {
                ids.Add(id);
            }
        }

        if (maxLength is not null && ids.Count > maxLength.GetValueOrDefault())
        {
            ids.RemoveRange(maxLength.GetValueOrDefault(), ids.Count - maxLength.GetValueOrDefault());
        }

        return ids.ToArray();
    }

    private List<int> EncodeUnigram(string text)
    {
        var ids = new List<int>();
        if (_unigramAddedTokens is null)
        {
            EncodeUnigramSegment(text.AsSpan(), ids);
            return ids;
        }

        int start = 0;
        int index = 0;
        while (index < text.Length)
        {
            int matchId = MatchAddedToken(text, index, out int matchLength);
            if (matchLength == 0)
            {
                index++;
                continue;
            }

            EncodeUnigramSegment(text.AsSpan(start, index - start), ids);
            ids.Add(matchId);
            index += matchLength;
            start = index;
        }

        EncodeUnigramSegment(text.AsSpan(start), ids);
        return ids;
    }

    private void EncodeUnigramSegment(ReadOnlySpan<char> segment, List<int> ids)
    {
        if (segment.IsEmpty)
        {
            return;
        }

        string text = segment.ToString();
        string encodeText = _unigramNormalizer is null ? text : _unigramNormalizer.Normalize(text);
        IReadOnlyList<EncodedToken> tokens = _tokenizer.EncodeToTokens(encodeText, out _, considerPreTokenization: true, considerNormalization: true);
        foreach (EncodedToken token in tokens)
        {
            if (_unigramVocab!.TryGetValue(token.Value, out int id))
            {
                ids.Add(id);
            }
        }
    }

    private int MatchAddedToken(string text, int index, out int matchLength)
    {
        matchLength = 0;
        if (_unigramAddedTokens is null)
        {
            return 0;
        }

        foreach (KeyValuePair<string, int> added in _unigramAddedTokens)
        {
            string content = added.Key;
            if (content.Length > 0 &&
                index + content.Length <= text.Length &&
                text.AsSpan(index, content.Length).SequenceEqual(content))
            {
                matchLength = content.Length;
                return added.Value;
            }
        }

        return 0;
    }

    private static Model2VecTokenizer CreateWordPiece(JsonElement root, JsonElement model, Dictionary<string, int> vocab, HashSet<int> unknownTokenIds)
    {
        JsonElement normalizer = root.GetProperty("normalizer");
        if (normalizer.GetProperty("type").GetString() != "BertNormalizer")
        {
            throw new NotSupportedException("WordPiece tokenizers require a BertNormalizer normalizer.");
        }

        if (root.TryGetProperty("pre_tokenizer", out JsonElement preTokenizer) &&
            preTokenizer.ValueKind != JsonValueKind.Null &&
            preTokenizer.GetProperty("type").GetString() != "BertPreTokenizer")
        {
            throw new NotSupportedException("WordPiece tokenizers require a BertPreTokenizer pre-tokenizer.");
        }

        string unkToken = model.TryGetProperty("unk_token", out JsonElement unk) ? unk.GetString() ?? "[UNK]" : "[UNK]";
        var options = new BertOptions
        {
            LowerCaseBeforeTokenization = ReadBoolean(normalizer, "lowercase", defaultValue: true),
            IndividuallyTokenizeCjk = ReadBoolean(normalizer, "handle_chinese_chars", defaultValue: true),
            RemoveNonSpacingMarks = ReadStripAccents(normalizer),
            ContinuingSubwordPrefix = ReadString(model, "continuing_subword_prefix", "##"),
            UnknownToken = unkToken,
            MaxInputCharsPerWord = ReadInt32(model, "max_input_chars_per_word", 100),
            ApplyBasicTokenization = true,
            SplitOnSpecialTokens = true,
            SpecialTokens = ReadSpecialTokens(root)
        };

        using var vocabStream = new MemoryStream(Encoding.UTF8.GetBytes(ToIdOrderedVocabulary(vocab)));
        BertTokenizer tokenizer = BertTokenizer.Create(vocabStream, options);
        return new Model2VecTokenizer(tokenizer, vocab.Count, unknownTokenIds, MedianTokenLength(vocab));
    }

    private static Model2VecTokenizer CreateBpe(JsonElement root, JsonElement model, Dictionary<string, int> vocab, HashSet<int> unknownTokenIds)
    {
        string? preTokenizerType = root.TryGetProperty("pre_tokenizer", out JsonElement preTokenizer) && preTokenizer.ValueKind != JsonValueKind.Null
            ? preTokenizer.GetProperty("type").GetString()
            : null;
        bool byteLevel = preTokenizerType == "ByteLevel";

        if (!byteLevel && preTokenizerType is not null)
        {
            throw new NotSupportedException($"BPE pre-tokenizer '{preTokenizerType}' is not supported.");
        }

        if (root.TryGetProperty("normalizer", out JsonElement normalizer) && normalizer.ValueKind != JsonValueKind.Null)
        {
            throw new NotSupportedException($"BPE normalizer '{normalizer.GetProperty("type").GetString()}' is not supported.");
        }

        var options = new BpeOptions(vocab)
        {
            Merges = ReadMerges(model.GetProperty("merges")),
            SpecialTokens = ReadSpecialTokens(root),
            UnknownToken = ReadUnknownToken(model, vocab),
            ContinuingSubwordPrefix = ReadNullableString(model, "continuing_subword_prefix"),
            EndOfWordSuffix = ReadNullableString(model, "end_of_word_suffix"),
            FuseUnknownTokens = ReadBoolean(model, "fuse_unk", defaultValue: false),
            ByteLevel = false
        };

        BpeTokenizer tokenizer = BpeTokenizer.Create(options);
        bool addPrefixSpace = byteLevel && ReadBoolean(preTokenizer, "add_prefix_space", defaultValue: false);
        return new Model2VecTokenizer(tokenizer, vocab.Count, unknownTokenIds, MedianTokenLength(vocab), byteLevelBpe: byteLevel, addPrefixSpace: addPrefixSpace);
    }

    private static Model2VecTokenizer CreateUnigram(string tokenizerJsonPath, JsonElement root, JsonElement model, Dictionary<string, int> vocab, HashSet<int> unknownTokenIds)
    {
        string directory = Path.GetDirectoryName(Path.GetFullPath(tokenizerJsonPath)) ?? Directory.GetCurrentDirectory();
        string? sentencePieceModel = new[] { "sentencepiece.model", "spiece.model", "tokenizer.model" }
            .Select(name => Path.Combine(directory, name))
            .FirstOrDefault(File.Exists);

        if (sentencePieceModel is not null)
        {
            using FileStream stream = File.OpenRead(sentencePieceModel);
            SentencePieceTokenizer fileTokenizer = SentencePieceTokenizer.Create(stream, addBeginningOfSentence: false, addEndOfSentence: false, ReadSpecialTokens(root));
            return new Model2VecTokenizer(fileTokenizer, vocab.Count, unknownTokenIds, MedianTokenLength(vocab), unigramVocab: vocab);
        }

        return CreateUnigramFromJson(root, model, vocab, unknownTokenIds);
    }

    private static Model2VecTokenizer CreateUnigramFromJson(JsonElement root, JsonElement model, Dictionary<string, int> vocab, HashSet<int> unknownTokenIds)
    {
        JsonElement vocabElement = model.GetProperty("vocab");
        if (vocabElement.ValueKind != JsonValueKind.Array)
        {
            throw new NotSupportedException("Unigram tokenizers require a [piece, score] vocabulary array.");
        }

        int count = vocabElement.GetArrayLength();
        var pieces = new string[count];
        var scores = new float[count];
        int id = 0;
        foreach (JsonElement entry in vocabElement.EnumerateArray())
        {
            pieces[id] = entry[0].GetString() ?? "";
            scores[id] = (float)entry[1].GetDouble();
            id++;
        }

        int unkId = model.TryGetProperty("unk_id", out JsonElement unk) && unk.ValueKind == JsonValueKind.Number
            ? unk.GetInt32()
            : throw new NotSupportedException("JSON-only Unigram tokenizers require a model.unk_id.");

        // model.unk_id defines the Unigram unknown piece, which must be removed before pooling
        // even when its piece string is not also discoverable via unk_token / "[UNK]".
        unknownTokenIds.Add(unkId);

        byte[] modelProto = SentencePieceModelProtoBuilder.Build(pieces, scores, unkId);
        using var protoStream = new MemoryStream(modelProto);
        SentencePieceTokenizer tokenizer = SentencePieceTokenizer.Create(protoStream, addBeginningOfSentence: false, addEndOfSentence: false);

        TextNormalizer? normalizer = root.TryGetProperty("normalizer", out JsonElement normalizerElement) && normalizerElement.ValueKind != JsonValueKind.Null
            ? TextNormalizer.Parse(normalizerElement)
            : null;

        return new Model2VecTokenizer(tokenizer, vocab.Count, unknownTokenIds, MedianTokenLength(vocab), unigramVocab: vocab, unigramNormalizer: normalizer, unigramAddedTokens: ReadSpecialTokens(root));
    }

    private static Dictionary<string, int> ReadVocabulary(JsonElement vocabElement)
    {
        var vocab = new Dictionary<string, int>(StringComparer.Ordinal);
        if (vocabElement.ValueKind == JsonValueKind.Array)
        {
            int id = 0;
            foreach (JsonElement entry in vocabElement.EnumerateArray())
            {
                vocab[entry[0].GetString() ?? ""] = id++;
            }

            return vocab;
        }

        foreach (JsonProperty property in vocabElement.EnumerateObject())
        {
            vocab.Add(property.Name, property.Value.GetInt32());
        }

        return vocab;
    }

    private static IReadOnlyDictionary<string, int> ReadSpecialTokens(JsonElement root)
    {
        var specialTokens = new Dictionary<string, int>(StringComparer.Ordinal);
        if (!root.TryGetProperty("added_tokens", out JsonElement addedTokens) || addedTokens.ValueKind != JsonValueKind.Array)
        {
            return specialTokens;
        }

        foreach (JsonElement token in addedTokens.EnumerateArray())
        {
            if (ReadBoolean(token, "special", defaultValue: false) &&
                token.TryGetProperty("content", out JsonElement content) &&
                token.TryGetProperty("id", out JsonElement id))
            {
                specialTokens[content.GetString() ?? ""] = id.GetInt32();
            }
        }

        return specialTokens;
    }

    private static HashSet<int> ReadUnknownTokenIds(JsonElement root, JsonElement model, Dictionary<string, int> vocab)
    {
        var ids = new HashSet<int>();
        string? unknownToken = ReadUnknownToken(model, vocab);
        if (unknownToken is not null && vocab.TryGetValue(unknownToken, out int vocabId))
        {
            ids.Add(vocabId);
        }

        if (root.TryGetProperty("added_tokens", out JsonElement addedTokens) && addedTokens.ValueKind == JsonValueKind.Array)
        {
            foreach (JsonElement token in addedTokens.EnumerateArray())
            {
                if (token.TryGetProperty("content", out JsonElement content) &&
                    content.GetString() == unknownToken &&
                    token.TryGetProperty("id", out JsonElement id))
                {
                    ids.Add(id.GetInt32());
                }
            }
        }

        return ids;
    }

    private static string? ReadUnknownToken(JsonElement model, Dictionary<string, int> vocab)
    {
        if (model.TryGetProperty("unk_token", out JsonElement unk) && unk.ValueKind == JsonValueKind.String)
        {
            return unk.GetString();
        }

        return vocab.ContainsKey("[UNK]") ? "[UNK]" : null;
    }

    private static string ToIdOrderedVocabulary(Dictionary<string, int> vocab)
    {
        string[] tokens = new string[vocab.Count];
        foreach ((string token, int id) in vocab)
        {
            if ((uint)id >= (uint)tokens.Length)
            {
                throw new InvalidDataException("Tokenizer vocabulary ids must be contiguous.");
            }

            tokens[id] = token;
        }

        return string.Join('\n', tokens);
    }

    private static string[] ReadMerges(JsonElement merges)
    {
        return merges.EnumerateArray()
            .Select(static merge => merge.ValueKind == JsonValueKind.Array
                ? string.Join(' ', merge.EnumerateArray().Select(static item => item.GetString()))
                : merge.GetString() ?? "")
            .ToArray();
    }

    private static bool ReadStripAccents(JsonElement normalizer)
    {
        bool lowercase = ReadBoolean(normalizer, "lowercase", defaultValue: true);
        if (!normalizer.TryGetProperty("strip_accents", out JsonElement stripAccents) || stripAccents.ValueKind == JsonValueKind.Null)
        {
            return lowercase;
        }

        return stripAccents.ValueKind == JsonValueKind.True;
    }

    private static bool ReadBoolean(JsonElement element, string propertyName, bool defaultValue)
    {
        return element.TryGetProperty(propertyName, out JsonElement property) && property.ValueKind is JsonValueKind.True or JsonValueKind.False
            ? property.GetBoolean()
            : defaultValue;
    }

    private static int ReadInt32(JsonElement element, string propertyName, int defaultValue)
    {
        return element.TryGetProperty(propertyName, out JsonElement property) && property.ValueKind == JsonValueKind.Number
            ? property.GetInt32()
            : defaultValue;
    }

    private static string ReadString(JsonElement element, string propertyName, string defaultValue)
    {
        return ReadNullableString(element, propertyName) ?? defaultValue;
    }

    private static string? ReadNullableString(JsonElement element, string propertyName)
    {
        return element.TryGetProperty(propertyName, out JsonElement property) && property.ValueKind == JsonValueKind.String
            ? property.GetString()
            : null;
    }

    private static int MedianTokenLength(Dictionary<string, int> vocab)
    {
        int[] lengths = vocab.Keys.Select(static token => token.Length).Order().ToArray();
        return lengths[lengths.Length / 2];
    }

    private static string ToByteLevelText(string text, bool addPrefixSpace)
    {
        if (addPrefixSpace && text.Length > 0 && !char.IsWhiteSpace(text[0]))
        {
            text = " " + text;
        }

        byte[] bytes = Encoding.UTF8.GetBytes(text);
        var builder = new StringBuilder(bytes.Length);
        foreach (byte b in bytes)
        {
            builder.Append(ByteLevelAlphabet[b]);
        }

        return builder.ToString();
    }

    private static char[] CreateByteLevelAlphabet()
    {
        var bytes = new List<int>();
        for (int i = 33; i <= 126; i++)
        {
            bytes.Add(i);
        }
        for (int i = 161; i <= 172; i++)
        {
            bytes.Add(i);
        }
        for (int i = 174; i <= 255; i++)
        {
            bytes.Add(i);
        }

        var chars = bytes.ToList();
        int next = 0;
        for (int b = 0; b < 256; b++)
        {
            if (!bytes.Contains(b))
            {
                bytes.Add(b);
                chars.Add(256 + next);
                next++;
            }
        }

        var map = new char[256];
        for (int i = 0; i < bytes.Count; i++)
        {
            map[bytes[i]] = (char)chars[i];
        }

        return map;
    }
}
