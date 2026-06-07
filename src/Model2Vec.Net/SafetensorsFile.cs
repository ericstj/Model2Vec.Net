using System.Buffers.Binary;
using System.Text.Json;

namespace Model2VecNet;

internal sealed class SafetensorsFile
{
    private readonly Dictionary<string, SafetensorTensor> _tensors;

    private SafetensorsFile(Dictionary<string, SafetensorTensor> tensors)
    {
        _tensors = tensors;
    }

    public static SafetensorsFile Load(string path)
    {
        byte[] file = File.ReadAllBytes(path);
        if (file.Length < sizeof(ulong))
        {
            throw new InvalidDataException("The safetensors file is too small.");
        }

        ulong headerLength = BinaryPrimitives.ReadUInt64LittleEndian(file.AsSpan(0, sizeof(ulong)));
        if (headerLength > int.MaxValue || checked(sizeof(ulong) + (int)headerLength) > file.Length)
        {
            throw new InvalidDataException("The safetensors header length is invalid.");
        }

        int headerStart = sizeof(ulong);
        int headerEnd = headerStart + (int)headerLength;
        using JsonDocument document = JsonDocument.Parse(file.AsMemory(headerStart, (int)headerLength));

        var tensors = new Dictionary<string, SafetensorTensor>(StringComparer.Ordinal);
        foreach (JsonProperty property in document.RootElement.EnumerateObject())
        {
            if (property.NameEquals("__metadata__"))
            {
                continue;
            }

            JsonElement value = property.Value;
            string dtype = value.GetProperty("dtype").GetString() ?? throw new InvalidDataException($"Tensor {property.Name} has no dtype.");
            long[] shape = value.GetProperty("shape").EnumerateArray().Select(static e => e.GetInt64()).ToArray();
            long[] offsets = value.GetProperty("data_offsets").EnumerateArray().Select(static e => e.GetInt64()).ToArray();
            if (offsets.Length != 2 || offsets[0] < 0 || offsets[1] < offsets[0])
            {
                throw new InvalidDataException($"Tensor {property.Name} has invalid offsets.");
            }

            long absoluteStart = headerEnd + offsets[0];
            long absoluteEnd = headerEnd + offsets[1];
            if (absoluteEnd > file.Length)
            {
                throw new InvalidDataException($"Tensor {property.Name} extends past the end of the file.");
            }

            tensors.Add(property.Name, new SafetensorTensor(dtype, shape, file.AsMemory((int)absoluteStart, checked((int)(absoluteEnd - absoluteStart)))));
        }

        return new SafetensorsFile(tensors);
    }

    public SafetensorTensor GetTensor(string name)
    {
        return _tensors.TryGetValue(name, out SafetensorTensor? tensor) ? tensor : throw new InvalidDataException($"Tensor '{name}' was not found.");
    }

    public SafetensorTensor? GetTensorOrDefault(string name)
    {
        return _tensors.TryGetValue(name, out SafetensorTensor? tensor) ? tensor : null;
    }
}

internal sealed class SafetensorTensor
{
    public SafetensorTensor(string dtype, long[] shape, ReadOnlyMemory<byte> data)
    {
        DType = dtype;
        Shape = shape;
        Data = data;
    }

    public string DType { get; }

    public long[] Shape { get; }

    public ReadOnlyMemory<byte> Data { get; }

    public float[] ToSingleArray()
    {
        long count = ElementCount();
        var result = new float[checked((int)count)];
        ReadOnlySpan<byte> span = Data.Span;
        switch (DType)
        {
            case "F32":
                for (int i = 0; i < result.Length; i++)
                {
                    result[i] = BitConverter.Int32BitsToSingle(BinaryPrimitives.ReadInt32LittleEndian(span.Slice(i * 4, 4)));
                }
                break;
            case "F16":
                for (int i = 0; i < result.Length; i++)
                {
                    result[i] = (float)BitConverter.UInt16BitsToHalf(BinaryPrimitives.ReadUInt16LittleEndian(span.Slice(i * 2, 2)));
                }
                break;
            case "F64":
                for (int i = 0; i < result.Length; i++)
                {
                    result[i] = (float)BitConverter.Int64BitsToDouble(BinaryPrimitives.ReadInt64LittleEndian(span.Slice(i * 8, 8)));
                }
                break;
            case "I8":
                for (int i = 0; i < result.Length; i++)
                {
                    result[i] = unchecked((sbyte)span[i]);
                }
                break;
            case "U8":
                for (int i = 0; i < result.Length; i++)
                {
                    result[i] = span[i];
                }
                break;
            default:
                throw new NotSupportedException($"Tensor dtype '{DType}' cannot be converted to Single.");
        }

        return result;
    }

    public int[] ToInt32Array()
    {
        long count = ElementCount();
        var result = new int[checked((int)count)];
        ReadOnlySpan<byte> span = Data.Span;
        switch (DType)
        {
            case "I64":
                for (int i = 0; i < result.Length; i++)
                {
                    result[i] = checked((int)BinaryPrimitives.ReadInt64LittleEndian(span.Slice(i * 8, 8)));
                }
                break;
            case "I32":
                for (int i = 0; i < result.Length; i++)
                {
                    result[i] = BinaryPrimitives.ReadInt32LittleEndian(span.Slice(i * 4, 4));
                }
                break;
            case "U32":
                for (int i = 0; i < result.Length; i++)
                {
                    result[i] = checked((int)BinaryPrimitives.ReadUInt32LittleEndian(span.Slice(i * 4, 4)));
                }
                break;
            default:
                throw new NotSupportedException($"Tensor dtype '{DType}' cannot be converted to Int32.");
        }

        return result;
    }

    private long ElementCount()
    {
        long count = 1;
        foreach (long dim in Shape)
        {
            count = checked(count * dim);
        }

        return count;
    }
}
