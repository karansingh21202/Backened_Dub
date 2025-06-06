# processor.py

import re
import torch
import numpy as np
import librosa
import soundfile as sf
import os
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import whisper
from TTS.api import TTS
import requests

# For environments with an event loop (e.g., Jupyter)
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass


# ─── Patched functions for TTS & torch.load ────────────────────────────────────
# (These were in your notebook to force TTS to run inside Colab’s loop.)

def patched_torch_load(*args, **kwargs):
    """
    In some environments (like Colab), torch.load must be patched
    so we don’t hit “Event loop is already running” errors.
    """
    if getattr(torch, "_original_load", None) is None:
        torch._original_load = torch.load  # stash original
    return torch._original_load(*args, **kwargs)


def patched_xtts_generate(self, cond_latents, text_inputs, **hf_generate_kwargs):
    """
    Patch the TTS model’s generate method so that it’s async-friendly
    (again, needed in Jupyter/Colab). In a pure Python backend, this
    patch is harmless but unnecessary; we keep it so that your code
    runs exactly as in Colab.
    """
    return self.tts_model.gpt.generate(cond_latents, text_inputs, **hf_generate_kwargs)


# Apply the patch immediately, so TTS.api’s internals get redirected.
torch.load = patched_torch_load


# ─── Helper: split text into overlapping chunks ────────────────────────────────
def advanced_split_text_to_chunks(text, max_chars=200, overlap_chars=20):
    """
    Splits a long string into chunks of up to `max_chars` characters,
    with `overlap_chars` overlap between consecutive chunks.
    This helps avoid cutting sentences mid-word when generating TTS.
    """
    words = text.strip().split()
    chunks = []
    current = ""
    for w in words:
        if len(current) + len(w) + 1 <= max_chars:
            current += ("" if current == "" else " ") + w
        else:
            chunks.append(current)
            # start new chunk with overlap
            overlap = " ".join(current.split()[-overlap_chars:])
            current = overlap + " " + w
    if current:
        chunks.append(current)
    return chunks


# ─── Helper: detect language from transcribed text ─────────────────────────────
def detect_language(text):
    """
    A minimal heuristic: if text has predominantly Devanagari (Hindi) characters,
    label as 'hi'; otherwise default to 'en'.
    You could call an external API, but this is a simple fallback.
    """
    devanagari_count = sum(1 for ch in text if "\u0900" <= ch <= "\u097F")
    return "hi" if devanagari_count > len(text) * 0.2 else "en"


# ─── Helper: merge multiple WAV chunks with a small crossfade ─────────────────
from scipy.signal import fftconvolve

def merge_audio_with_crossfade(chunk_files, output_file, crossfade_ms=100):
    """
    Given a list of file paths (WAV chunks), read each one, then
    overlap‐add them with `crossfade_ms` milliseconds of crossfading
    so that the final output is one continuous WAV.
    """
    if not chunk_files:
        raise ValueError("No chunk files to merge")

    # Read first chunk
    base_y, base_sr = librosa.load(chunk_files[0], sr=None)
    output = base_y.copy()

    crossfade_samples = int((crossfade_ms / 1000) * base_sr)
    for idx in range(1, len(chunk_files)):
        y, sr = librosa.load(chunk_files[idx], sr=base_sr)
        # crossfade between `output[-crossfade_samples:]` and `y[:crossfade_samples]`
        fade_out = np.linspace(1.0, 0.0, crossfade_samples)
        fade_in = np.linspace(0.0, 1.0, crossfade_samples)

        # Apply crossfade to end of `output` and start of `y`
        output[-crossfade_samples:] = (
            output[-crossfade_samples:] * fade_out
            + y[:crossfade_samples] * fade_in
        )
        output = np.concatenate((output, y[crossfade_samples:]))

    # Save merged result
    sf.write(output_file, output, base_sr)
    return output_file


# ─── Helper: analyze audio for energy/pitch (used for visualization) ──────────
def analyze_audio(audio_path):
    """
    Returns a dict containing:
      • waveform (numpy array)
      • sampling rate
      • short-time energy curve
      • smoothed energy curve
      • pitch contour (approximate)
    (Optional: your notebook may have used this to visualize or select chunks.)
    """
    y, sr = librosa.load(audio_path, sr=None)
    # Energy
    frame_length = int(0.025 * sr)
    hop_length = int(0.010 * sr)
    energy = np.array(
        [
            np.sum(np.abs(y[i : i + frame_length] ** 2))
            for i in range(0, len(y), hop_length)
        ]
    )
    # Smooth energy
    smoothed = savgol_filter(energy, 51, 3)

    # Pitch (using librosa’s piptrack)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_contour = [
        np.mean(p[p > 0]) if np.any(p > 0) else 0.0 for p in pitches.T
    ]

    return {
        "waveform": y,
        "sr": sr,
        "energy": energy,
        "smoothed_energy": smoothed,
        "pitch_contour": pitch_contour,
    }


# ─── Helper: visualize speech characteristics (optional) ──────────────────────
def visualize_speech_characteristics(audio_path, save_path=None):
    """
    Plots waveform, energy, and pitch. If save_path is given, saves a PNG.
    """
    data = analyze_audio(audio_path)
    y, sr = data["waveform"], data["sr"]
    energy = data["energy"]
    smoothed = data["smoothed_energy"]
    pitch = data["pitch_contour"]

    fig, axes = plt.subplots(3, 1, figsize=(10, 8))

    # Waveform
    axes[0].plot(np.linspace(0, len(y) / sr, len(y)), y)
    axes[0].set_title("Waveform")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_xlabel("Time (s)")

    # Energy
    times_e = np.linspace(0, len(y) / sr, len(energy))
    axes[1].plot(times_e, energy, label="Raw Energy")
    axes[1].plot(times_e, smoothed, label="Smoothed", linewidth=2)
    axes[1].legend()
    axes[1].set_title("Short-Time Energy")

    # Pitch
    times_p = np.linspace(0, len(y) / sr, len(pitch))
    axes[2].plot(times_p, pitch)
    axes[2].set_title("Pitch Contour")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Frequency (Hz)")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    plt.close(fig)
    return save_path


# ─── Helper: translate text via Google Translate API ─────────────────────────
def google_translate_text(api_key, text, source_language, target_language):
    """
    Calls Google’s translation endpoint. Returns translated text.
    Make sure you have enabled the Google Translate API for your key.
    """
    url = "https://translation.googleapis.com/language/translate/v2"
    params = {
        "key": api_key,
        "q": text,
        "source": source_language,
        "target": target_language,
        "format": "text",
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    return data["data"]["translations"][0]["translatedText"]


# ─── Main function: voice_dub ─────────────────────────────────────────────────
def voice_dub(
    input_audio_path: str,
    output_file: str = "output_dubbed.wav",
    chunk_size: int = 200,
    visualize: bool = False,
    use_advanced_chunking: bool = True,
):
    """
    1) Loads input WAV (input_audio_path).
    2) Runs Whisper → transcribes language + text.
    3) Splits text into chunks (overlapping) via advanced_split_text_to_chunks.
    4) Generates TTS chunks with voice cloning, using the same speaker WAV.
    5) Merges chunks with crossfade.
    6) Translates source_text → target_language via Google Translate.
    7) Saves final dubbed audio at output_file.
    Returns a dict with metadata and path to `output_file`.
    """

    output_dir = os.path.dirname(os.path.abspath(output_file))
    os.makedirs(output_dir, exist_ok=True)

    print("Initializing models...")
    # Load Whisper model conditionally (GPU vs CPU)
    whisper_model = (
        whisper.load_model("large")
        if torch.cuda.is_available()
        else whisper.load_model("medium")
    )

    # Replace with your actual Google Translation API key.
    API_KEY = "AIzaSyAeS0eXbmRipDiiV9mOoO8djLwZitvcYNY"

    # Load TTS model (XTTS) for bilingual voice cloning
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=torch.cuda.is_available())
    # Patch XTTS’s generate method
    tts.tts_to_file  # ensure loaded
    tts.tts_to_file.__globals__["patched_xtts_generate"] = patched_xtts_generate

    print("Analyzing original audio for voice characteristics...")
    if visualize:
        visualize_speech_characteristics(input_audio_path)

    print("Transcribing with Whisper...")
    result = whisper_model.transcribe(input_audio_path)
    source_text = result["text"].strip()
    source_lang = result["language"]

    print(f"Detected source language: {source_lang}, text length: {len(source_text)} chars.")

    # Decide target language: if source is English, dub to Hindi; else to English
    target_language = "hi" if source_lang.startswith("en") else "en"
    print(f"Target language for TTS will be: {target_language}")

    # 1) Split into chunks
    if use_advanced_chunking:
        chunks = advanced_split_text_to_chunks(source_text, max_chars=chunk_size, overlap_chars=20)
    else:
        # naive split every `chunk_size` characters
        chunks = [
            source_text[i : i + chunk_size] for i in range(0, len(source_text), chunk_size)
        ]

    print(f"Total chunks: {len(chunks)}")

    # 2) Generate TTS for each chunk
    audio_chunk_files = []
    print("Generating TTS with voice cloning...")
    for i, chunk in enumerate(tqdm(chunks, desc="TTS chunks")):
        chunk_file = os.path.join(output_dir, f"chunk_{i}.wav")
        tts.tts_to_file(
            text=chunk,
            speaker_wav=input_audio_path,
            language=target_language,
            file_path=chunk_file,
        )
        audio_chunk_files.append(chunk_file)

    # 3) Merge with crossfade
    merged_path = os.path.join(output_dir, f"merged_{os.path.basename(output_file)}")
    print("Merging audio chunks with crossfade...")
    merge_audio_with_crossfade(audio_chunk_files, merged_path, crossfade_ms=100)

    # 4) Translate the text (so you can return target_text)
    print("Translating text via Google Translate API...")
    target_text = google_translate_text(API_KEY, source_text, source_lang, target_language)

    # 5) Copy or rename merged file to `output_file`
    os.replace(merged_path, output_file)

    print(f"Voice dubbing complete! Output saved to: {output_file}")
    return {
        "source_text": source_text,
        "source_language": source_lang,
        "target_text": target_text,
        "target_language": target_language,
        "output_file": output_file,
    }
