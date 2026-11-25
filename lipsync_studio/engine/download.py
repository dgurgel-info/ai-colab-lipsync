"""Baixa os pesos do InfiniteTalk (versão fp8). Roda DENTRO do venv do modelo.

    python -m lipsync_studio.engine.download --weights /content/lipsync_engine/weights

Baixa só o que o modo fp8 usa (~32 GB em vez de ~95 GB):

* Wan-AI/Wan2.1-I2V-14B-480P: configuração, VAE, CLIP e tokenizadores (a DiT vem do arquivo fp8)
* MeiGen-AI/InfiniteTalk: DiT + InfiniteTalk em fp8 e o codificador de texto T5 em fp8
* TencentGameMate/chinese-wav2vec2-base: codificador de áudio (``model.safetensors`` da refs/pr/1)
"""

from __future__ import annotations

import argparse
import os
import sys

DOWNLOADS = (
    # (repositório, pasta local, arquivos ou padrões, revisão)
    ("Wan-AI/Wan2.1-I2V-14B-480P", "Wan2.1-I2V-14B-480P",
     ["config.json", "Wan2.1_VAE.pth", "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
      "google/umt5-xxl/*", "xlm-roberta-large/*"], None),
    ("MeiGen-AI/InfiniteTalk", "InfiniteTalk",
     ["quant_models/infinitetalk_single_fp8.safetensors", "quant_models/infinitetalk_single_fp8.json",
      "quant_models/t5_fp8.safetensors", "quant_models/t5_map_fp8.json"], None),
    ("TencentGameMate/chinese-wav2vec2-base", "chinese-wav2vec2-base",
     ["config.json", "preprocessor_config.json"], None),
    ("TencentGameMate/chinese-wav2vec2-base", "chinese-wav2vec2-base", ["model.safetensors"], "refs/pr/1"),
)

# Arquivos que precisam existir no final (o worker não sobe sem eles).
REQUIRED = (
    "Wan2.1-I2V-14B-480P/config.json",
    "Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth",
    "Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
    "Wan2.1-I2V-14B-480P/google/umt5-xxl/tokenizer.json",
    "Wan2.1-I2V-14B-480P/xlm-roberta-large/tokenizer.json",
    "InfiniteTalk/quant_models/infinitetalk_single_fp8.safetensors",
    "InfiniteTalk/quant_models/infinitetalk_single_fp8.json",
    "InfiniteTalk/quant_models/t5_fp8.safetensors",
    "InfiniteTalk/quant_models/t5_map_fp8.json",
    "chinese-wav2vec2-base/config.json",
    "chinese-wav2vec2-base/preprocessor_config.json",
    "chinese-wav2vec2-base/model.safetensors",
)


def missing(weights: str):
    return [f for f in REQUIRED if not os.path.isfile(os.path.join(weights, f))]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    args = parser.parse_args(argv)

    from huggingface_hub import snapshot_download

    os.makedirs(args.weights, exist_ok=True)
    for repo, folder, patterns, revision in DOWNLOADS:
        target = os.path.join(args.weights, folder)
        print(f"⬇️ {repo}{' @ ' + revision if revision else ''}: {', '.join(patterns)}", flush=True)
        snapshot_download(repo, local_dir=target, allow_patterns=patterns, revision=revision)
        print(f"   ✓ {target}", flush=True)

    lost = missing(args.weights)
    if lost:
        print("❌ Arquivos que não chegaram:\n  " + "\n  ".join(lost), flush=True)
        return 1
    total = sum(os.path.getsize(os.path.join(args.weights, f)) for f in REQUIRED)
    print(f"✅ Pesos prontos ({total / 1e9:.1f} GB).", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
