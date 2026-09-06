# Filelists

Every manifest in this folder is a *template*. Replace the rows with your own
data; do not change the column contract without also changing `data.columns`
in your config.

| File | Format | Use for |
|---|---|---|
| `template_single_speaker.txt` | `audio\|text` | LJSpeech and any single-speaker corpus (Experiment 1) |
| `template_vctk.txt` | `audio\|speaker\|text` | VCTK 0.92 (flac, `_mic1`) or the older wav48 layout |
| `template_libritts.txt` | `audio\|speaker\|text` | LibriTTS -- true casing and punctuation |
| `template_librispeech.txt` | `audio\|speaker\|text` | LibriSpeech -- ALL CAPS, no punctuation; prefer LibriTTS |
| `template_emotion.csv` | CSV with header | expressive / emotional TTS |
| `template_full.jsonl` | JSONL | anything with extra per-utterance fields |
| `ljspeech_*.txt` | `audio\|text` | written by `scripts/prepare_dataset.py --dataset ljspeech` |

Paths are relative to `data.root` in the config. Absolute paths also work.

Tell the loader which columns a pipe file has:

```yaml
data:
  columns: [audio, speaker, emotion, text]
  delimiter: "|"
```

Full field reference: [`docs/DATA_FORMAT.md`](../docs/DATA_FORMAT.md).


Generate real filelists instead of editing these by hand:

```bash
python scripts/prepare_dataset.py --dataset vctk     --root data/VCTK-Corpus-0.92
python scripts/prepare_dataset.py --dataset libritts --root data/LibriTTS --subsets train-clean-100
```
