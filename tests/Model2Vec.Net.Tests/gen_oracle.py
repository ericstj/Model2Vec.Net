import json
import os
from model2vec import StaticModel

MODELS = [
    (
        "minishlab/potion-base-2M",
        "oracle_potion_base_2m.json",
        [
            "Hello, how are you doing today?",
            "The quick brown fox jumps over the lazy dog.",
            "Model2Vec turns token embeddings into fast sentence vectors.",
            "Bonjour, comment allez-vous aujourd'hui?",
            "Hola, ¿cómo estás hoy?",
            "Привет, как у тебя дела сегодня?",
            "こんにちは、お元気ですか。",
            "你好，今天过得怎么样？",
            "Numbers 123 456, punctuation, and MIXED case!",
            "",
        ],
    ),
    (
        "Jarbas/ovos-model2vec-intents-distilroberta-base-ca-v2",
        "oracle_distilroberta_base_ca_v2.json",
        [
            "Hello world from a byte-level BPE tokenizer.",
            "Olá mundo! Acentos: café, ação, coração.",
            "Català: una frase amb l·l, accents i signes d’interrogació?",
            "English, português, català, and numbers 123 456.",
            "Привет мир — byte fallback handles UTF-8 bytes.",
            "你好，今天过得怎么样？",
            "",
        ],
    ),
]

for model_name, oracle_file, sentences in MODELS:
    local_name = "distilroberta-base-ca-v2" if "distilroberta" in model_name else model_name.rsplit("/", 1)[1]
    local_path = os.path.join(os.path.dirname(__file__), "models", local_name)
    model_source = local_path if os.path.exists(os.path.join(local_path, "model.safetensors")) else model_name
    model = StaticModel.from_pretrained(model_source)
    embeddings = model.encode(sentences).tolist()
    out = {
        "model": model_name,
        "dimension": int(model.dim),
        "normalize": bool(model.normalize),
        "cases": [
            {"text": text, "embedding": [float(x) for x in embedding]}
            for text, embedding in zip(sentences, embeddings)
        ],
    }
    path = os.path.join(os.path.dirname(__file__), oracle_file)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"wrote {path} with {len(sentences)} cases, dim={model.dim}")
