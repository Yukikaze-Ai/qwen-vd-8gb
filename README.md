# Describing a voice and getting it fast

Qwen3-TTS VoiceDesign invents a speaker from a written description instead of copying one from a reference clip. On
a single 8 GB consumer GPU it ran at RTF 1.41 — slower than playback — and two settings bring it to **RTF 0.98**
across five different descriptions, using plain Hugging Face Transformers and no serving stack.

## Result

Same sentence, same card (RTX 3060 Ti 8 GB), bf16:

| Configuration | Steady RTF | Peak VRAM | First render |
|---|---|---|---|
| As shipped | 1.41 | 4.09 GiB | — |
| Code predictor compiled, `recompile_limit=8` (default) | 1.19 | 4.06 GiB | 75 s |
| Code predictor compiled, `recompile_limit=64` | 0.98 | 4.07 GiB | 84 s |
| **+ talker's inner transformer compiled** | **0.78** | 4.1 GiB | 100–250 s |

Five descriptions — low-voiced young woman, bright teenage girl, composed thirty-something, whispered, elderly —
gave 0.97, 0.98, 0.97, 0.97, 0.97. The cost is uniform.

Compiling the talker as well — specifically its inner transformer, not the wrapper — improves the steady state
substantially: 0.77–0.80 on four-second renders, 0.92 on a seven-second one.

**It does not guarantee below 1.0, and an earlier draft of this file claimed that.** A later run of the same
recipe on the same machine measured 1.03–1.19, with the GPU at normal clocks and temperature, while a rerun
minutes later was back at 0.98. The variance tracks what else is using the card. Across every run, the predictor-
only configuration sat at 0.97–1.02 and the two-compilation configuration at 0.77–1.19: better, not bounded.

Profiling explains where it goes: 99% of a render is in the autoregressive loop, 1% in the vocoder. Compiling the
vocoder would be pointless; compiling both model halves inside the loop is not.

## The two settings

```python
torch._dynamo.config.specialize_int = False
torch._dynamo.config.recompile_limit = 64
torch.compiler.set_stance("default")

# the 1.7B talker: compile the inner transformer, not the wrapper
inner = model.model.talker.model
inner.forward = torch.compile(inner.forward, mode="default", dynamic=True)

# the 5-layer code predictor
p = model.model.talker.code_predictor
p.forward = torch.compile(p.forward, mode="default", dynamic=True)
```

Compiling `talker.forward` itself fails with `UncapturedHigherOrderOpError`: that wrapper calls the predictor's
`generate()`, which contains data-dependent control flow. The inner transformer has no such call and compiles.

**`recompile_limit` is where this note earns its keep.** It defaults to eight. The code predictor's sequence length
grows every step, so the limit is exhausted inside the first utterance and torch then falls back to eager for every
later shape — with one warning and no error. Measuring only "compilation succeeded" reports a win that stopped
applying after a few shapes. Raising the limit took the steady state from 1.19 to 0.98.

`mode="default"` rather than `reduce-overhead`: CUDA graph capture fixes output buffers, and the generation loop
reads them across steps, which fails. `dynamic=True` because the sequence length changes every frame.

## Does compiling change the voice?

Twenty Chinese sentences, rendered uncompiled and compiled, scored with CER (Paraformer) and UTMOS (SpeechMOS):

| Variant | CER % | UTMOS |
|---|---|---|
| Uncompiled | 1.41 | 3.25 |
| Compiled | 2.07 | 3.23 |

The 0.65-point CER gap is not a regression. Four sentences moved and they split evenly: two where the compiled
variant lost or gained a sentence-initial particle the recogniser had previously caught or missed, two where the
uncompiled one did. Leading interjections are the recogniser's weak spot, not the audio's.


## A controlled run

Twenty renders of one sentence in a single process, GPU utilisation sampled before each: median RTF **0.80**,
P10 0.80, P90 0.83, **0 of 20 above 1.0**, utilisation 18-43%. The spread that reached 1.19 came from renders where
something else was using the card; on an otherwise idle machine this configuration is stable.

## Cost

The first render in a process pays 57–84 s of compilation. A service must gate readiness on a completed synthesis,
not on weights having loaded, or its first user waits a minute and a half.

Peak VRAM is 4.07 GiB: comfortable on 8 GB, feasible on 6 GB. Quantisation was not attempted — at this footprint it
would buy nothing on the target card.

## What is not claimed

- One machine, one model revision, single runs per configuration.
- No listening test. UTMOS predicts naturalness; it cannot tell whether a voice *is the voice the description asked
  for*. Whether ten descriptions yield ten genuinely different speakers is a human judgement not made here.
- No speaker-similarity score: VoiceDesign has no reference clip, so there is nothing to compute a cosine against.
- Only the code predictor was compiled; the talker stayed eager and CUDA graphs were not pursued.

## Reproducing

`recipe.py` loads the VoiceDesign checkpoint, applies the three settings, renders the five descriptions, and prints
steady RTF plus peak memory while writing the audio.

```bash
QWEN_VD_DIR=./qwen3-tts-vd-1.7b python recipe.py
```

## Paper

`paper/` holds the same material as a 3-page technical report: `main.tex` is the source, `main.pdf` the compiled
output, `tectonic main.tex` rebuilds it.

## License

Apache-2.0 for the code, CC BY 4.0 for this text.
