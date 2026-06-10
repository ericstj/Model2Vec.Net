using System.Buffers.Binary;
using System.Text;

namespace Model2VecNet;

/// <summary>
/// Serializes a minimal SentencePiece <c>ModelProto</c> protobuf for a Unigram tokenizer
/// reconstructed from a Hugging Face tokenizer.json that ships without a <c>.model</c> file.
/// Microsoft.ML.Tokenizers' <c>SentencePieceTokenizer</c> consumes this to provide the Unigram
/// Viterbi decoder and Metaspace handling; normalization (including the precompiled charsmap)
/// is applied separately before encoding, so the emitted normalizer spec carries no charsmap.
/// </summary>
internal static class SentencePieceModelProtoBuilder
{
    private enum PieceType
    {
        Normal = 1,
        Unknown = 2,
    }

    public static byte[] Build(IReadOnlyList<string> pieces, IReadOnlyList<float> scores, int unkId)
    {
        if (pieces.Count != scores.Count)
        {
            throw new ArgumentException("Piece and score counts must match.");
        }

        if ((uint)unkId >= (uint)pieces.Count)
        {
            throw new InvalidDataException("Unigram unk_id is outside the vocabulary.");
        }

        var stream = new MemoryStream();

        for (int i = 0; i < pieces.Count; i++)
        {
            PieceType type = i == unkId ? PieceType.Unknown : PieceType.Normal;
            WritePiece(stream, pieces[i], scores[i], type);
        }

        WriteTrainerSpec(stream, unkId, unkPiece: pieces[unkId]);
        WriteNormalizerSpec(stream);

        return stream.ToArray();
    }

    private static void WritePiece(Stream stream, string piece, float score, PieceType type)
    {
        var message = new MemoryStream();
        WriteString(message, fieldNumber: 1, piece);
        WriteFixed32(message, fieldNumber: 2, BitConverter.SingleToUInt32Bits(score));
        WriteVarintField(message, fieldNumber: 3, (ulong)type);
        WriteLengthDelimited(stream, fieldNumber: 1, message.ToArray());
    }

    private static void WriteTrainerSpec(Stream stream, int unkId, string unkPiece)
    {
        // SentencePieceTokenizer requires valid bos/eos ids; point them at the unknown slot so
        // the reconstructed model never marks a real piece as a control token.
        var message = new MemoryStream();
        WriteVarintField(message, fieldNumber: 3, (ulong)TrainerModelType.Unigram); // model_type
        WriteVarintField(message, fieldNumber: 40, (ulong)unkId);                   // unk_id
        WriteVarintField(message, fieldNumber: 41, (ulong)unkId);                   // bos_id
        WriteVarintField(message, fieldNumber: 42, (ulong)unkId);                   // eos_id
        WriteSignedVarintField(message, fieldNumber: 43, -1);                       // pad_id (disabled)
        WriteString(message, fieldNumber: 45, unkPiece);                            // unk_piece
        WriteString(message, fieldNumber: 46, unkPiece);                            // bos_piece
        WriteString(message, fieldNumber: 47, unkPiece);                            // eos_piece
        WriteLengthDelimited(stream, fieldNumber: 2, message.ToArray());
    }

    private static void WriteNormalizerSpec(Stream stream)
    {
        var message = new MemoryStream();
        WriteString(message, fieldNumber: 1, "identity"); // name; charsmap applied separately
        WriteVarintField(message, fieldNumber: 3, 1);      // add_dummy_prefix
        WriteVarintField(message, fieldNumber: 4, 0);      // remove_extra_whitespaces (handled by the JSON chain)
        WriteVarintField(message, fieldNumber: 5, 1);      // escape_whitespaces
        WriteLengthDelimited(stream, fieldNumber: 3, message.ToArray());
    }

    private enum TrainerModelType
    {
        Unigram = 1,
    }

    private static void WriteString(Stream stream, int fieldNumber, string value)
        => WriteLengthDelimited(stream, fieldNumber, Encoding.UTF8.GetBytes(value));

    private static void WriteLengthDelimited(Stream stream, int fieldNumber, byte[] payload)
    {
        WriteTag(stream, fieldNumber, wireType: 2);
        WriteVarint(stream, (ulong)payload.Length);
        stream.Write(payload, 0, payload.Length);
    }

    private static void WriteFixed32(Stream stream, int fieldNumber, uint value)
    {
        WriteTag(stream, fieldNumber, wireType: 5);
        Span<byte> buffer = stackalloc byte[4];
        BinaryPrimitives.WriteUInt32LittleEndian(buffer, value);
        stream.Write(buffer);
    }

    private static void WriteVarintField(Stream stream, int fieldNumber, ulong value)
    {
        WriteTag(stream, fieldNumber, wireType: 0);
        WriteVarint(stream, value);
    }

    private static void WriteSignedVarintField(Stream stream, int fieldNumber, int value)
    {
        WriteTag(stream, fieldNumber, wireType: 0);
        WriteVarint(stream, unchecked((ulong)(long)value));
    }

    private static void WriteTag(Stream stream, int fieldNumber, int wireType)
        => WriteVarint(stream, (ulong)((fieldNumber << 3) | wireType));

    private static void WriteVarint(Stream stream, ulong value)
    {
        while (value >= 0x80)
        {
            stream.WriteByte((byte)(value | 0x80));
            value >>= 7;
        }

        stream.WriteByte((byte)value);
    }
}
