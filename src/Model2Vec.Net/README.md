# Model2Vec.Net

A pure-managed C# (`net10.0`) port of [MinishLab Model2Vec](https://github.com/MinishLab/model2vec)
static-embedding **inference**. Load a Model2Vec model folder
(`model.safetensors` + `tokenizer.json` + `config.json`) and compute sentence
embeddings with no Python, no native libraries, and no ONNX.

> Unofficial, independent port. Not affiliated with or endorsed by MinishLab. See
> the project repository for full attribution and third-party notices.

## Features

- Pure C# (`net10.0`), no native dependency, no P/Invoke.
- Reads the safetensors format directly (`F32`, `F16`, `F64`, `I8`, `U8`).
- Supports Model2Vec `embeddings` and Sentence Transformers `embedding.weight` tensors,
  plus Model2Vec vocabulary quantization (`weights`/`mapping`).
- Hugging Face tokenizers via [`Microsoft.ML.Tokenizers`](https://github.com/dotnet/machinelearning-tokenizers):
  WordPiece, byte-level BPE, and SentencePiece-backed Unigram.
- SIMD-accelerated scaling and normalization via `System.Numerics.Tensors`.
- Output verified against the Python `model2vec` package (tolerance `1e-4`).

## Installation

```pwsh
dotnet add package Model2Vec.Net
```

## Usage

```csharp
using Model2VecNet;

var model = Model2VecModel.Load(@"C:\models\potion-base-2M");

float[] embedding = model.Encode("The quick brown fox jumps over the lazy dog.");

Console.WriteLine(model.Dimension);   // e.g. 256
Console.WriteLine(embedding.Length);  // == Dimension
```

Batch encoding:

```csharp
float[][] embeddings = model.Encode([
    "First sentence",
    "Second sentence",
]);
```

`Model2VecModel` is immutable after loading and safe to share across threads.

## Getting a model

Models are published on Hugging Face and are **not** bundled with this package.
Download a model folder (containing `model.safetensors`, `tokenizer.json`, and
`config.json`) and pass its path to `Model2VecModel.Load`. For example:

- [`minishlab/potion-base-2M`](https://huggingface.co/minishlab/potion-base-2M)
- [`minishlab/potion-base-8M`](https://huggingface.co/minishlab/potion-base-8M)

Any Model2Vec model in the standard folder layout will load.

## Upstream / attribution

- Model2Vec (reference implementation): https://github.com/MinishLab/model2vec
- Microsoft.ML.Tokenizers: https://github.com/dotnet/machinelearning-tokenizers

## License

MIT. See the project repository for the license and third-party notices:
https://github.com/ericstj/Model2Vec.Net
