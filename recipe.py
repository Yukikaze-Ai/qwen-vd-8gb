"""VoiceDesign on one consumer GPU: the two settings that get it under RTF 1.0.

Point QWEN_VD_DIR at the VoiceDesign checkpoint, then run:

    QWEN_VD_DIR=./qwen3-tts-vd-1.7b python recipe.py

Optional:
    TEXTS      semicolon-separated sentences (default: one)
    INSTRUCT   semicolon-separated voice descriptions, rendered in order
    OUT        output directory, default ./out
    LIMIT      torch._dynamo.config.recompile_limit, default 64

It prints steady RTF and peak memory per description. The first render pays compilation
(57-84 s on the card this was measured on) and is reported separately.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

MODEL_DIR = os.environ["QWEN_VD_DIR"]
OUT = Path(os.environ.get("OUT", "./out"))
LIMIT = int(os.environ.get("LIMIT", "64"))
TEXT = os.environ.get("TEXT", "你来啦，我等你好久了。今天外面下雨，记得带伞。")

DEFAULT_INSTRUCTS = [
    "一位二十岁出头的年轻女性，声音偏低、语速偏慢，语气温和但略带倦意",
    "十六七岁的少女，声音明亮清脆，语速偏快，带一点雀跃的尾音",
    "三十岁左右的成熟女性，中低音，咬字清晰、语速平稳，语气克制而温柔",
    "声音很轻、贴近耳边的气声，尾音带一点沙哑，像困了还在强撑着说话",
    "六十岁左右的长者，声音温厚、语速很慢，句尾习惯性地上扬一点点",
]
INSTRUCTS = [s for s in os.environ.get("INSTRUCT", "").split(";") if s.strip()] or DEFAULT_INSTRUCTS


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    from qwen_tts import Qwen3TTSModel

    # Condition 1: stop specialising on the incrementing step counter.
    torch._dynamo.config.specialize_int = False
    # Condition 2 (the decisive one): the predictor's sequence length grows every frame, and the
    # default limit of 8 is exhausted inside the first utterance -- after which torch falls back to
    # eager with a single warning and no error.
    torch._dynamo.config.recompile_limit = LIMIT
    torch.compiler.set_stance("default")

    model = Qwen3TTSModel.from_pretrained(MODEL_DIR, device_map="cuda:0", dtype=torch.bfloat16)

    # Condition 3: the 1.7B talker's inner transformer. Its outer forward calls the predictor's
    # generate(), which contains data-dependent control flow and cannot be captured -- compiling the
    # outer wrapper raises UncapturedHigherOrderOpError. The inner model compiles fine and is where
    # the time is: profiling put 99% of a render in the autoregressive loop and 1% in the vocoder.
    talker_inner = getattr(model.model.talker, "model", None)
    if talker_inner is not None:
        talker_inner.forward = torch.compile(talker_inner.forward, mode="default", dynamic=True)

    predictor = model.model.talker.code_predictor
    # mode="default": CUDA graphs fix output buffers and the generate loop reads them across steps.
    # dynamic=True: the predictor's sequence length changes every frame.
    predictor.forward = torch.compile(predictor.forward, mode="default", dynamic=True)

    torch.cuda.reset_peak_memory_stats()
    steady: list[float] = []
    for i, instruct in enumerate(INSTRUCTS):
        t0 = time.perf_counter()
        with torch.inference_mode():
            wavs, sr = model.generate_voice_design(
                text=TEXT, instruct=instruct, language="chinese", non_streaming_mode=True
            )
        elapsed = time.perf_counter() - t0
        audio = np.asarray(wavs[0], dtype=np.float32)
        secs = len(audio) / sr
        rtf = elapsed / max(secs, 1e-6)
        path = OUT / f"voice_{i:02d}.wav"
        sf.write(path, audio, sr, subtype="PCM_16")
        tag = "  (includes compilation)" if i == 0 else ""
        print(f"  voice_{i:02d}  {secs:4.2f}s audio  synth {elapsed:5.2f}s  RTF {rtf:5.2f}{tag}")
        print(f"    {instruct}")
        if i:
            steady.append(rtf)

    peak = torch.cuda.max_memory_allocated() / 1024**3
    if steady:
        print(f"\n  steady RTF: mean {sum(steady)/len(steady):.2f}  "
              f"best {min(steady):.2f}  worst {max(steady):.2f}  (first render excluded)")
    print(f"  peak VRAM {peak:.2f} GiB   output {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
