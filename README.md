# Model2Vec.Net

A pure-managed C# port of [MinishLab Model2Vec](https://github.com/MinishLab/model2vec)
static-embedding **inference**. It loads Model2Vec model folders containing
`model.safetensors`, `tokenizer.json`, and `config.json`, then computes sentence
embeddings without Python, native libraries, or ONNX.

Model2Vec inference is:

1. tokenize with the model tokenizer, without special tokens;
2. remove unknown-token ids;
3. truncate to `maxLength` tokens (default: 512);
4. gather embedding rows for token ids;
5. apply optional vocabulary-quantization weights if present;
6. mean-pool over tokens, returning zeros for empty token lists;
7. L2-normalize when `config.json` has `"normalize": true`.

## Features

- Pure C# (`net10.0`), no native dependency, no P/Invoke.
- Reads the safetensors format directly.
- Supports Model2Vec `embeddings` tensors and Sentence Transformers
  `embedding.weight` tensors.
- Supports `F32`, `F16`, `F64`, `I8`, and `U8` embedding tensors.
- Supports Model2Vec vocabulary-quantization `weights` and `mapping` tensors.
- Uses `Microsoft.ML.Tokenizers` for Hugging Face WordPiece and byte-level BPE tokenizers.
- SIMD-accelerated scaling and normalization via `System.Numerics.Tensors`.
- Implements `Microsoft.Extensions.AI` `IEmbeddingGenerator<string, Embedding<float>>`
  for use in the .NET AI ecosystem (RAG, vector stores, semantic search).

## Tokenizer support

The library parses `tokenizer.json`, dispatches by `model.type`, and constructs the
corresponding `Microsoft.ML.Tokenizers` tokenizer:

- `WordPiece`: `BertTokenizer`, including `BertNormalizer` settings such as
  lowercase, accent stripping, CJK splitting, unknown token, continuation prefix,
  and maximum input characters per word.
- `BPE`: `BpeTokenizer`. Byte-level BPE tokenizers are supported, including
  GPT-2/Roberta byte-to-unicode preprocessing and `add_prefix_space`.
- `Unigram`: supported when a SentencePiece `.model` file is present alongside
  `tokenizer.json`; Hugging Face JSON-only Unigram vocabularies require follow-up
  support because `Microsoft.ML.Tokenizers` 2.0.0 loads SentencePiece from `.model`.

All tokenizers encode without special tokens, remove unknown-token ids before
pooling, and apply Model2Vec pre-truncation and final `maxLength` truncation.

## Scope: inference only

Model2Vec.Net implements the **inference** half of Model2Vec — loading a distilled
static model and encoding text. It deliberately does **not** include model
**distillation or training**:

- Distilling a new static model from a teacher sentence-transformer (forward-passing
  the vocabulary, PCA dimensionality reduction, and Zipf/SIF weighting).
- The `tokenlearn` corpus post-training step and classifier-head training
  (`model2vec.train`, `model2vec.distill`).

**Why these are out of scope:** distillation and training require running a full
transformer encoder and an autodiff/optimizer training loop. In .NET that means
taking a **native deep-learning dependency** (ONNX Runtime or libtorch), which would
break this package's defining property: pure-managed with **no native dependency**.
Distillation is also a one-time, offline, GPU-friendly step — you produce a model
once with the upstream Python tooling and load the resulting `model.safetensors`
here. If managed distillation is ever needed it belongs in a **separate** package
built on a DL runtime, keeping this inference core small and dependency-free.

## Usage

```csharp
using Model2VecNet;

var model = Model2VecModel.Load(@"C:\models\potion-base-2M");
float[] embedding = model.Encode("The quick brown fox jumps over the lazy dog.");

Console.WriteLine(model.Dimension);
Console.WriteLine(embedding.Length);
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

Model files are published on Hugging Face and are not bundled in this repository.
The test suite downloads:

- `minishlab/potion-base-2M`
- `Jarbas/ovos-model2vec-intents-distilroberta-base-ca-v2`
- `model.safetensors`
- `tokenizer.json`
- `config.json`

## Building and testing

```pwsh
dotnet build -c Release
dotnet test -c Release
```

The oracle tests compare .NET embeddings against Python `model2vec` outputs for
the WordPiece and BPE test models with element-wise tolerance `1e-4`.

## Benchmarks

The BenchmarkDotNet suite is under `bench\Model2Vec.Net.Benchmarks` and covers
single short-text encode, single long-text encode, batch encode, and model load.
Place `potion-base-2M` under the benchmark project's `models` folder or set
`MODEL2VEC_POTION_BASE_2M`.

## License

MIT — see [LICENSE](LICENSE). See [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md)
for attribution.
