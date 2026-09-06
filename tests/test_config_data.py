"""Config, text frontend, audio features and manifest parsing."""

from pathlib import Path

import pytest
import torch

from mint_tts.config import Config, load_config, merge, parse_value
from mint_tts.data.audio import AudioConfig, load_wav, mel_spectrogram, save_wav
from mint_tts.data.manifest import read_manifest
from mint_tts.evaluation.metrics import cer, mel_cepstral_distortion, wer

ROOT = Path(__file__).resolve().parents[1]


# -- config -------------------------------------------------------------------
def test_config_inheritance_and_override():
    base = load_config(ROOT / "configs" / "base.yaml")
    exp = load_config(ROOT / "configs" / "exp0_dense.yaml")
    assert exp.model.encoder.routing == "fixed"
    assert exp.model.encoder.max_steps == base.model.encoder.max_steps  # inherited
    assert exp.loss.compute.enabled is False


def test_cli_overrides_are_typed():
    cfg = load_config(ROOT / "configs" / "base.yaml",
                      ["train.batch_size=4", "train.amp=true", "model.d_model=48",
                       "log.run_name=zzz"])
    assert cfg.train.batch_size == 4 and isinstance(cfg.train.batch_size, int)
    assert cfg.train.amp is True
    assert cfg.model.d_model == 48
    assert cfg.log.run_name == "zzz"


def test_parse_value():
    assert parse_value("true") is True
    assert parse_value("null") is None
    assert parse_value("3") == 3
    assert parse_value("[1, 2]") == [1, 2]
    assert parse_value("hello") == "hello"


def test_merge_is_deep():
    a = Config({"x": {"y": 1, "z": 2}})
    b = {"x": {"y": 9}}
    assert merge(a, b).x.to_dict() == {"y": 9, "z": 2}


# -- audio --------------------------------------------------------------------
def test_mel_shape_and_roundtrip(tmp_path):
    ac = AudioConfig()
    wav = torch.sin(2 * 3.14159 * 220 * torch.arange(ac.sample_rate) / ac.sample_rate) * 0.5
    mel = mel_spectrogram(wav, ac)
    assert mel.shape[0] == ac.n_mels
    # HiFi-GAN-style manual reflect padding gives exactly len(wav)//hop frames
    assert mel.shape[1] == ac.sample_rate // ac.hop_length
    assert torch.isfinite(mel).all()

    path = tmp_path / "x.wav"
    save_wav(path, wav, ac.sample_rate)
    back = load_wav(path, ac)
    assert abs(back.numel() - wav.numel()) <= 1


def test_mel_rejects_unnormalised_audio():
    with pytest.raises(ValueError):
        mel_spectrogram(torch.full((1000,), 5.0), AudioConfig())


# -- manifests ----------------------------------------------------------------
def test_read_pipe_manifest():
    recs = read_manifest(ROOT / "filelists" / "template_single_speaker.txt", ["audio", "text"])
    assert len(recs) >= 2
    assert recs[0].audio.endswith(".wav") and recs[0].text
    assert recs[0].speaker == "default" and recs[0].emotion == "neutral"


@pytest.mark.parametrize("name, speakers", [
    ("template_vctk.txt", {"p225", "p226", "p227", "p228"}),
    ("template_libritts.txt", {"19", "26", "32"}),
    ("template_librispeech.txt", {"19", "26"}),
])
def test_read_multispeaker_manifests(name, speakers):
    recs = read_manifest(ROOT / "filelists" / name, ["audio", "speaker", "text"])
    assert {r.speaker for r in recs} == speakers
    assert all(r.audio and r.text for r in recs)


def test_read_emotion_csv():
    recs = read_manifest(ROOT / "filelists" / "template_emotion.csv")
    assert {"happy", "sad", "angry", "neutral", "surprised"} <= {r.emotion for r in recs}


def test_read_jsonl():
    recs = read_manifest(ROOT / "filelists" / "template_full.jsonl")
    assert recs[0].lang == "en"
    assert any(r.style for r in recs)


def test_manifest_rejects_missing_columns(tmp_path):
    bad = tmp_path / "bad.txt"
    bad.write_text("|no audio path\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_manifest(bad, ["audio", "text"])


# -- metrics ------------------------------------------------------------------
def test_wer_cer():
    assert wer("the cat sat", "the cat sat") == 0.0
    assert wer("the cat sat", "the dog sat") == pytest.approx(1 / 3)
    assert cer("abc", "abc") == 0.0
    assert cer("abc", "abd") == pytest.approx(1 / 3)


def test_mcd_is_zero_for_identical_mels():
    mel = torch.randn(80, 50)
    assert mel_cepstral_distortion(mel, mel) == pytest.approx(0.0, abs=1e-4)
    assert mel_cepstral_distortion(mel, mel + 1.0) > 0.0


# -- multi-speaker configs ----------------------------------------------------
@pytest.mark.parametrize("name", ["vctk_token", "libritts_token"])
def test_multi_speaker_configs(name):
    cfg = load_config(ROOT / "configs" / f"{name}.yaml")
    assert cfg.data.columns == ["audio", "speaker", "text"]
    assert str(cfg.model.n_speakers) == "auto"


@pytest.mark.parametrize("name", ["frontend_char", "frontend_ipa", "frontend_arpabet"])
def test_frontend_configs_differ_only_in_input_type(name):
    cfg = load_config(ROOT / "configs" / f"{name}.yaml")
    base = load_config(ROOT / "configs" / "exp2_token.yaml")
    assert cfg.text.input_type == name.split("_")[1]
    assert cfg.model.to_dict() == base.model.to_dict()
    assert cfg.data.preprocessed_dir != base.data.preprocessed_dir
