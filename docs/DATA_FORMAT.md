# Data format

## What you must provide

Audio plus a transcript. Nothing else. Durations are learned by the internal
aligner, pitch and energy are extracted automatically, and the compute labels
`C*` are generated later by `scripts/compute_curve.py` from the model's own
behaviour. There is no manual annotation step anywhere in this pipeline.

## Manifest formats

Three formats are accepted; pick whichever matches the corpus you already have.
Templates for each live in [`filelists/`](../filelists/).

### 1. Pipe filelist (`.txt`)

```
wavs/LJ001-0001.wav|Printing, in the only sense with which we are concerned,
wavs/LJ001-0002.wav|differs from most if not from all the arts and crafts.
```

Column order comes from the config, so any arrangement works:

```yaml
data:
  columns: [audio, text]                      # single speaker
  # columns: [audio, speaker, text]           # multi-speaker
  # columns: [audio, speaker, emotion, text]  # expressive
  delimiter: "|"
```

### 2. CSV with a header (`.csv`)

Column names are read from the header, so no config change is needed.

```csv
audio,speaker,emotion,text
wavs/0001.wav,spk01,neutral,"The meeting has been moved to three o'clock."
wavs/0002.wav,spk01,happy,"We got the grant!"
```

### 3. JSONL (`.jsonl`)

One object per line. Best when you carry extra fields.

```json
{"audio": "wavs/0001.wav", "text": "Hello there.", "speaker": "spk01", "emotion": "neutral", "lang": "en", "style": "conversational"}
```

## Fields

| Field | Required | Default | Meaning |
|---|---|---|---|
| `audio` | yes | — | path to the waveform, relative to `data.root` (absolute also works) |
| `text` | yes | — | raw transcript; normalisation happens in the frontend |
| `speaker` | no | `default` | speaker id. Set `model.n_speakers > 1` to enable the embedding |
| `emotion` | no | `neutral` | emotion/style label. Set `model.n_emotions > 1` to enable the embedding |
| `lang` | no | `en` | language tag, reserved for multilingual work |
| `style` | no | `""` | free-form style prompt, carried through but not yet consumed by the model |
| `uid` | no | audio stem | unique id; feature files are named after it |

Templates for all of these live in [`filelists/`](../filelists/).

## Splits

`scripts/prepare_dataset.py --dataset ljspeech` writes `<prefix>_{train,val,test,all}.txt`.
For your own corpus, produce the same three files. `scripts/preprocess.py`
looks for `<prefix>_<split>.txt`, derived from `data.manifest` unless you pass
`--filelist-prefix`.

Keep the split honest: the test set should never appear in `data.train_index`.

## What preprocessing writes

```
data/preprocessed/<dataset>/
├── mel/<uid>.npy       (n_mels, T) float32 log-mel
├── pitch/<uid>.npy     (T,) frame-level F0
├── energy/<uid>.npy    (T,) frame-level energy
├── train.jsonl         index: tokens, token strings, word map, lengths, paths
├── val.jsonl
├── test.jsonl
├── symbols.json        the vocabulary, derived from this corpus
├── stats.json          pitch/energy normalisation, speaker and emotion maps, vocab size
└── *_errors.json       per-file failures, if any
```

An index row:

```json
{
  "uid": "LJ001-0001",
  "audio": "data/LJSpeech-1.1/wavs/LJ001-0001.wav",
  "text": "The record is broken by the record broker.",
  "clean_text": "the record is broken by the record broker.",
  "tokens": [1, 20, 8, 5, 4, ...],
  "token_strings": ["<bos>", "t", "h", "e", "<sp>", ...],
  "word_ids": [0, 0, 0, 0, 0, 1, ...],
  "words": ["the", "record", "is", "broken", "by", "the", "record", "broker"],
  "n_frames": 812, "n_tokens": 44, "duration_sec": 9.43,
  "mel": "...", "pitch": "...", "energy": "..."
}
```

`token_strings`, `word_ids` and `words` exist purely so the complexity
heatmaps can be labelled with real words instead of indices.

After running `scripts/compute_curve.py --write-labels`, rows gain a
`c_star` field — the generated minimum-compute supervision.

## Audio requirements

* Any sample rate and channel count: audio is downmixed and resampled to
  `audio.sample_rate` (22050 by default) on the fly.
* Peak-normalised to 0.95 and, with `audio.trim_silence: true`, leading and
  trailing silence is trimmed with an energy threshold.
* If you change any of `sample_rate`, `n_fft`, `hop_length`, `n_mels`, `fmin`
  or `fmax`, your mels no longer match standard HiFi-GAN checkpoints. Either
  retrain the vocoder or keep the defaults.

## Text frontend

```yaml
text:
  input_type: ipa        # char | ipa | arpabet
  lowercase: true
  add_bos_eos: true
  add_word_boundary: true
  phonemizer:
    language: en-us
    with_stress: true
  skip_normalisation_steps: []   # e.g. [acronyms] to leave capitals alone
```

| Backend | Needs | Notes |
|---|---|---|
| `char` | nothing | the model sees letters; every ambiguity survives |
| `ipa` | `phonemizer` + `espeakng-loader` | espeak-ng arrives via pip, no system package |
| `arpabet` | `g2p_en` + `nltk` | downloads CMUdict and the POS tagger on first use |

Run `python scripts/inspect_frontend.py` before choosing. Neither phonemiser
reliably resolves homographs, which is exactly why the choice is an
experimental variable here rather than a settled default. If a backend is
missing the frontend warns loudly and falls back to characters rather than
failing silently.

### Normalisation

Normalisation runs before phonemisation, as an ordered pipeline (currency
before bare numbers, dates before ordinals, phones before digit groups). It
covers numbers, ordinals, years, currency, percentages, temperatures, units,
dates, times, phone numbers, emails, URLs, street addresses, unit designators,
titles, general abbreviations and acronyms.

When an expansion looks wrong, `TextNormalizer().trace(text)` prints the text
after every step so you can see which rule fired.

### Vocabulary

Symbol tables are **derived from your corpus**, not hard-coded, so IPA works
for any language espeak supports without anyone maintaining a phone list.
Preprocessing writes `symbols.json` next to the indices; training loads it and
freezes it, so val/test can never introduce symbols the model was not trained
on. Change `text.input_type` and you must preprocess into a *different*
`preprocessed_dir` — the vocabularies are not interchangeable.

## Multi-speaker

```yaml
data:
  columns: [audio, speaker, text]
model:
  n_speakers: auto      # read from stats.json at train time
```

`auto` resolves to the real speaker count from the preprocessed corpus, so the
config cannot silently disagree with the data. The same applies to
`n_emotions`. `scripts/prepare_dataset.py` handles VCTK (both the 0.92 flac
layout and the older wav48 one), LibriTTS and LibriSpeech; pass
`--speaker-disjoint` to hold out whole speakers instead of utterances, which is
the zero-shot-speaker experiment rather than the standard one.
