# arXiv submission sheet — Qwen3-TTS VoiceDesign

Paste these fields into the arXiv form. Separate submission from the MOSS one; the two papers do not cite each other.

## Title

```
Describing a Voice and Getting It Fast: Qwen3-TTS VoiceDesign at RTF 0.78 on One 8 GB Consumer GPU
```

## Authors

```
Hongyun Du
```

## Affiliation

```
Independent Researcher
```

## Abstract

```
Most text-to-speech models copy a voice from a reference clip; fewer can invent one from a written description.
Qwen3-TTS VoiceDesign does the second, and that capability is the reason to consider it, but on a consumer card it
ran at RTF 1.41 -- slower than playback, so unfit for conversation. We report three settings that bring it to
RTF 0.78 on four-second utterances and 0.92 on seven-second ones, with five different descriptions, on a single
GeForce RTX 3060 Ti using Hugging Face Transformers and no serving stack. Profiling first put 99% of a render in the
autoregressive loop and 1% in the vocoder, so the work is inside the loop: the five-layer code predictor, whose
sequence length changes every frame, and the 1.7B talker's inner transformer, which can be compiled where its
wrapper cannot. The decisive parameter is one nobody sets by default -- torch._dynamo.config.recompile_limit is 8,
exhausted inside the first utterance, after which compilation silently stops applying. We also report that these
gains are not bounded: later runs of the same recipe measured up to RTF 1.19 depending on what else used the card.
Costs are 100-250 s of compilation on the first render, peak 4.07 GiB, and no measurable change in intelligibility
or predicted naturalness over 20 sentences.
```

## Comments

```
4 pages, 3 tables. Code and reproduction script: https://github.com/Yukikaze-Ai/qwen-vd-8gb
```

## Category and license

```
Primary category  cs.SD (Sound)
Cross-list        cs.LG   (optional)
License           CC BY 4.0
```

## Files to upload

```
paper/main.tex        LaTeX source -- arXiv compiles it and shows a preview
```

## Order of operations

1. Create the public repository and confirm the URL resolves.
2. Replace the placeholder URL in Comments if the repository name differs.
3. Start a new submission, paste the fields, upload main.tex.
4. Check the arXiv-generated PDF preview -- it is built from source, not from our local build.
5. Submit.

## What the reviewer will notice

- The report does not claim a new algorithm. It claims that an existing implementation reaches a usable speed on one
  consumer card, with the exact settings and a negative result about the stability of that speed.
- No listening test. UTMOS predicts naturalness and cannot tell whether a voice matches its description; the paper
  says so.
- The description-to-voice mapping is unreliable without explicit gender, which is reported as a finding rather than
  hidden.
