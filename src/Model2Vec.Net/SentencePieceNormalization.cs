using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace Model2VecNet;

/// <summary>
/// Applies a Hugging Face tokenizer.json normalizer chain in managed code so that JSON-only
/// SentencePiece (Unigram) tokenizers can be normalized identically to the reference
/// implementation before the SentencePiece model performs Metaspace pre-tokenization and
/// Unigram decoding.
/// </summary>
internal abstract class TextNormalizer
{
    public abstract string Normalize(string text);

    public static TextNormalizer Parse(JsonElement normalizer)
    {
        string? type = normalizer.TryGetProperty("type", out JsonElement typeElement) ? typeElement.GetString() : null;
        switch (type)
        {
            case "Sequence":
                JsonElement steps = normalizer.GetProperty("normalizers");
                var children = new List<TextNormalizer>(steps.GetArrayLength());
                foreach (JsonElement step in steps.EnumerateArray())
                {
                    children.Add(Parse(step));
                }

                return new SequenceNormalizer(children);

            case "Precompiled":
                string charsMap = normalizer.GetProperty("precompiled_charsmap").GetString()
                    ?? throw new InvalidDataException("Precompiled normalizer is missing precompiled_charsmap.");
                return new PrecompiledNormalizer(Convert.FromBase64String(charsMap));

            case "Replace":
                return ReplaceNormalizer.Create(normalizer);

            case "Strip":
                return new StripNormalizer(
                    stripLeft: !normalizer.TryGetProperty("strip_left", out JsonElement left) || left.ValueKind != JsonValueKind.False,
                    stripRight: !normalizer.TryGetProperty("strip_right", out JsonElement right) || right.ValueKind != JsonValueKind.False);

            case "Lowercase":
                return new LowercaseNormalizer();

            default:
                throw new NotSupportedException($"Unigram normalizer type '{type}' is not supported.");
        }
    }
}

internal sealed class SequenceNormalizer(IReadOnlyList<TextNormalizer> normalizers) : TextNormalizer
{
    private readonly IReadOnlyList<TextNormalizer> _normalizers = normalizers;

    public override string Normalize(string text)
    {
        foreach (TextNormalizer normalizer in _normalizers)
        {
            text = normalizer.Normalize(text);
        }

        return text;
    }
}

internal sealed class LowercaseNormalizer : TextNormalizer
{
    public override string Normalize(string text) => text.ToLowerInvariant();
}

internal sealed class StripNormalizer(bool stripLeft, bool stripRight) : TextNormalizer
{
    private readonly bool _stripLeft = stripLeft;
    private readonly bool _stripRight = stripRight;

    public override string Normalize(string text)
    {
        if (_stripLeft && _stripRight)
        {
            return text.Trim();
        }

        if (_stripLeft)
        {
            return text.TrimStart();
        }

        return _stripRight ? text.TrimEnd() : text;
    }
}

internal sealed class ReplaceNormalizer : TextNormalizer
{
    private readonly string _content;
    private readonly string? _literal;
    private readonly Regex? _regex;

    private ReplaceNormalizer(string? literal, Regex? regex, string content)
    {
        _literal = literal;
        _regex = regex;
        _content = content;
    }

    public static ReplaceNormalizer Create(JsonElement normalizer)
    {
        string content = normalizer.GetProperty("content").GetString() ?? "";
        JsonElement pattern = normalizer.GetProperty("pattern");

        if (pattern.TryGetProperty("String", out JsonElement literal))
        {
            return new ReplaceNormalizer(literal.GetString() ?? "", regex: null, content);
        }

        if (pattern.TryGetProperty("Regex", out JsonElement regex))
        {
            string regexPattern = regex.GetString() ?? throw new InvalidDataException("Replace normalizer has a null Regex pattern.");
            return new ReplaceNormalizer(literal: null, new Regex(regexPattern, RegexOptions.CultureInvariant), content);
        }

        throw new NotSupportedException("Replace normalizer requires a String or Regex pattern.");
    }

    public override string Normalize(string text)
    {
        if (_regex is not null)
        {
            return _regex.Replace(text, _content);
        }

        return _literal!.Length == 0 ? text : text.Replace(_literal, _content, StringComparison.Ordinal);
    }
}

/// <summary>
/// Applies a SentencePiece precompiled character map (the normalizer charsmap embedded in
/// tokenizer.json) using a Darts double-array trie, matching SentencePiece's longest-match
/// replacement. Port of the precompiled-charsmap handling in Microsoft.ML.Tokenizers'
/// SentencePieceNormalizer.
/// </summary>
internal sealed class PrecompiledNormalizer : TextNormalizer
{
    private const int MaxTrieResults = 32;

    private readonly uint[] _trie;
    private readonly byte[] _normalized;

    public PrecompiledNormalizer(ReadOnlySpan<byte> blob)
    {
        if (blob.Length <= sizeof(uint))
        {
            throw new InvalidDataException("Precompiled charsmap blob is too small.");
        }

        uint trieSize = (uint)(blob[0] | (blob[1] << 8) | (blob[2] << 16) | (blob[3] << 24));
        if (trieSize < sizeof(uint) || trieSize % sizeof(uint) != 0)
        {
            throw new InvalidDataException("Precompiled charsmap trie size must be a non-zero multiple of 4 bytes.");
        }

        if (trieSize >= blob.Length - sizeof(uint))
        {
            throw new InvalidDataException("Precompiled charsmap trie size exceeds the blob size.");
        }

        ReadOnlySpan<byte> body = blob.Slice(sizeof(uint));
        int unitCount = (int)(trieSize / sizeof(uint));
        _trie = new uint[unitCount];
        for (int i = 0; i < unitCount; i++)
        {
            int offset = i * sizeof(uint);
            _trie[i] = (uint)(body[offset] | (body[offset + 1] << 8) | (body[offset + 2] << 16) | (body[offset + 3] << 24));
        }

        _normalized = body.Slice((int)trieSize).ToArray();
    }

    public override string Normalize(string text)
    {
        if (text.Length == 0)
        {
            return text;
        }

        byte[] input = Encoding.UTF8.GetBytes(text);
        var output = new List<byte>(input.Length);
        int index = 0;
        while (index < input.Length)
        {
            int consumed = NormalizePrefix(input.AsSpan(index), out int valueOffset, out bool replaced);
            if (replaced)
            {
                for (int i = valueOffset; i < _normalized.Length && _normalized[i] != 0; i++)
                {
                    output.Add(_normalized[i]);
                }
            }
            else
            {
                for (int i = 0; i < consumed; i++)
                {
                    output.Add(input[index + i]);
                }
            }

            index += consumed;
        }

        return Encoding.UTF8.GetString(output.ToArray());
    }

    private int NormalizePrefix(ReadOnlySpan<byte> input, out int valueOffset, out bool replaced)
    {
        valueOffset = 0;

        int longestLength = 0;
        int longestValue = 0;

        Span<int> lengths = stackalloc int[MaxTrieResults];
        Span<int> values = stackalloc int[MaxTrieResults];
        int count = CommonPrefixSearch(input, lengths, values);
        if (count > lengths.Length)
        {
            count = lengths.Length;
        }

        for (int k = 0; k < count; k++)
        {
            if (longestLength == 0 || lengths[k] > longestLength)
            {
                longestLength = lengths[k];
                longestValue = values[k];
            }
        }

        if (longestLength == 0)
        {
            replaced = false;
            return Utf8SequenceLength(input[0]);
        }

        replaced = true;
        valueOffset = longestValue;
        return longestLength;
    }

    private int CommonPrefixSearch(ReadOnlySpan<byte> key, Span<int> lengths, Span<int> values)
    {
        int numResults = 0;
        int nodePos = 0;
        uint unit = _trie[nodePos];
        nodePos ^= (int)Offset(unit);

        for (int i = 0; i < key.Length; i++)
        {
            nodePos ^= key[i];
            unit = _trie[nodePos];
            if (Label(unit) != key[i])
            {
                return numResults;
            }

            nodePos ^= (int)Offset(unit);
            if (HasLeaf(unit))
            {
                if (numResults < lengths.Length)
                {
                    values[numResults] = (int)Value(_trie[nodePos]);
                    lengths[numResults] = i + 1;
                }

                numResults++;
            }
        }

        return numResults;
    }

    private static bool HasLeaf(uint unit) => ((unit >> 8) & 1) == 1;

    private static uint Value(uint unit) => unit & ((1U << 31) - 1);

    // Bit 31 is intentionally retained so leaf units yield an out-of-range label that never
    // matches a byte key, which is how darts-clone distinguishes leaves during traversal.
    private static uint Label(uint unit) => unit & ((1U << 31) | 0xFF);

    private static uint Offset(uint unit) => (unit >> 10) << (int)((unit & (1U << 9)) >> 6);

    private static int Utf8SequenceLength(byte b)
    {
        if (b < 0x80)
        {
            return 1;
        }

        if ((b & 0xE0) == 0xC0)
        {
            return 2;
        }

        if ((b & 0xF0) == 0xE0)
        {
            return 3;
        }

        if ((b & 0xF8) == 0xF0)
        {
            return 4;
        }

        return 1;
    }
}
