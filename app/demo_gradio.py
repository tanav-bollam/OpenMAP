"""Gradio demo: upload or record a clip, see the top-3 genres and the mel-spectrogram.

    python app/demo.py                       # uses the run with the best test macro-F1
    python app/demo.py --run outputs/cnn     # force a run (cnn or mert)

Preprocessing matches training: native decode -> resample (22.05 kHz mel for the CNN,
24 kHz waveform for MERT) -> 10 s windows (5 s hop) -> average probabilities.
Clips longer than 30 s use their first 30 s; clips shorter than 10 s are zero-padded.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

print("Starting the demo: loading libraries and the model. This can take 1-3 minutes the first time after a restart; "
      "please wait (don't press Ctrl+C) until it says 'Running on local URL'.", flush=True)

import gradio as gr  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import audio  # noqa: E402
from models import build_model  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def patch_gradio_upload_race() -> None:
    """Work around a Gradio bug on Windows that floods the terminal with
    'h11 LocalProtocolError: Too little data for declared Content-Length' after a microphone recording.

    Gradio stores an upload at <cache>/<sha256 of content>/<name>. The recorder uploads the same
    recording twice, and shutil.move() onto the existing file falls back (on Windows) to copy-over,
    which truncates the file while the first request is still serving it. Same hash folder means the
    same bytes, so when the destination already exists we just drop the redundant temp file."""
    import os
    import shutil

    import gradio.routes as gr_routes

    if getattr(gr_routes, "_upload_race_patched", False):
        return

    def safe_move(files, destinations):
        for src, dest in zip(files, destinations, strict=False):
            if os.path.exists(dest):
                _remove_quietly(src)
                continue
            try:
                shutil.move(src, dest)
            except OSError:
                if os.path.exists(dest):  # another request created it first
                    _remove_quietly(src)
                else:
                    raise

    def _remove_quietly(path):
        try:
            os.remove(path)
        except OSError:
            pass

    gr_routes.move_uploaded_files_to_cache = safe_move
    gr_routes._upload_race_patched = True


def pick_run() -> Path:
    best, best_f1 = None, -1.0
    for d in sorted((ROOT / "outputs").glob("*")):
        f = d / "eval_test.json"
        if (d / "best.pt").exists() and f.exists() and not d.name.endswith("_quick"):
            f1 = json.loads(f.read_text())["macro_f1"]
            if f1 > best_f1:
                best, best_f1 = d, f1
    if best is None:
        raise SystemExit("No evaluated run found: run src/evaluate.py first, or pass --run outputs/<name>")
    return best


class Predictor:
    def __init__(self, run_dir: Path):
        ckpt = torch.load(run_dir / "best.pt", map_location=DEVICE, weights_only=False)
        self.cfg, self.genres = ckpt["config"], ckpt["genres"]
        self.name = f"{run_dir.name} ({self.cfg['model']['name']})"
        self.model = build_model(self.cfg, len(self.genres)).to(DEVICE).eval()
        self.model.load_state_dict(ckpt["model"])
        self.kind = self.cfg["data"]["kind"]
        if self.kind == "mel":
            from utils import mel_stats_path
            s = json.loads(mel_stats_path(self.cfg["data"]["subset"]).read_text())
            self.mean, self.std = s["mean"], s["std"]
        elif self.kind == "embedding":
            from extract_mert import load_mert
            self.mert = load_mert(self.cfg["mert"], DEVICE).half()
        elif self.kind == "wave":
            pass  # e.g. AST: the model computes its own features from 16 kHz waveform windows
        else:
            raise SystemExit(f"demo does not support data.kind={self.kind!r}")

    @torch.no_grad()
    def __call__(self, path: str):
        y, sr = audio.decode_any(path)
        y = y[: int(30 * sr)]
        if len(y) < 10 * sr:
            y = np.pad(y, (0, int(10 * sr) - len(y)))
        y22 = audio.resample(y, sr, audio.MEL_SR)
        mel = audio.logmel(y22)
        if self.kind == "mel":
            crop = self.cfg["data"].get("crop_frames", audio.WINDOW_FRAMES)
            n_win = max(1, int(round((mel.shape[1] - crop) / (5 * audio.MEL_SR / audio.HOP))) + 1)
            starts = audio.window_starts(mel.shape[1], crop, n_win)
            x = np.stack([mel[:, s:s + crop] for s in starts])
            x = torch.from_numpy((x - self.mean) / self.std)[:, None].to(DEVICE)
            with torch.autocast(DEVICE.type, dtype=torch.bfloat16, enabled=DEVICE.type == "cuda"):
                probs = torch.softmax(self.model(x).float(), -1).mean(0)
        elif self.kind == "wave":
            d = self.cfg["data"]
            sr_t = d["sample_rate"]
            yw = audio.resample(y, sr, sr_t)
            crop = int(d.get("crop_seconds", 10.0) * sr_t)
            if len(yw) < crop:
                yw = np.pad(yw, (0, crop - len(yw)))
            n_win = max(1, int(round((len(yw) - crop) / (5 * sr_t))) + 1)
            starts = audio.window_starts(len(yw), crop, n_win)
            x = torch.from_numpy(np.stack([yw[s:s + crop] for s in starts])).to(DEVICE)
            with torch.autocast(DEVICE.type, dtype=torch.bfloat16, enabled=DEVICE.type == "cuda"):
                probs = torch.softmax(self.model(x).float(), -1).mean(0)
        else:
            from extract_mert import embed_chunks, pool_track
            mc = self.cfg["mert"]
            y24 = audio.resample(y, sr, mc["sample_rate"])
            c = int(mc["chunk_seconds"] * mc["sample_rate"])
            n = max(1, len(y24) // c)
            chunks = torch.from_numpy(np.pad(y24, (0, max(0, n * c - len(y24))))[: n * c].reshape(n, c)).to(DEVICE)
            emb = pool_track(embed_chunks(self.mert, chunks), chunks)[None]
            probs = torch.softmax(self.model(emb).float(), -1)[0]
        return probs.cpu().numpy(), mel


def make_ui(pred: Predictor):
    patch_gradio_upload_race()

    def run(path):
        if path is None:
            raise gr.Error("Upload or record an audio clip first.")
        try:
            probs, mel = pred(path)
        except Exception as e:  # noqa: BLE001
            # Full traceback goes to the terminal and logs/demo_errors.log; the page shows the message.
            import traceback
            tb = traceback.format_exc()
            print(tb, flush=True)
            try:
                (ROOT / "logs").mkdir(exist_ok=True)
                with open(ROOT / "logs" / "demo_errors.log", "a", encoding="utf-8") as f:
                    f.write(f"--- {path}\n{tb}\n")
            except OSError:
                pass
            raise gr.Error(f"Could not process this audio ({type(e).__name__}: {e})")
        top = np.argsort(-probs)[:3]
        fig, ax = plt.subplots(figsize=(6, 2.2))
        ax.barh([pred.genres[i] for i in top][::-1], [probs[i] for i in top][::-1], color="#3b6fb6")
        ax.set_xlim(0, 1)
        ax.set_xlabel("probability")
        for k, i in enumerate(top[::-1]):
            ax.text(min(probs[i] + 0.01, 0.85), k, f"{probs[i]:.1%}", va="center", fontsize=9)
        fig.tight_layout()
        fig2, ax2 = plt.subplots(figsize=(6, 3))
        ax2.imshow(mel, origin="lower", aspect="auto", cmap="magma")
        ax2.set_xlabel("frame"), ax2.set_ylabel("mel bin"), ax2.set_title("log-mel spectrogram")
        fig2.tight_layout()
        return {pred.genres[i]: float(probs[i]) for i in top}, fig, fig2

    return gr.Interface(
        fn=run,
        # format=None: pass recordings through untouched. Gradio's default (format="wav") converts any
        # non-WAV file on the server with ffmpeg, which fails on machines without it. We decode ourselves.
        inputs=gr.Audio(sources=["upload", "microphone"], type="filepath", format=None,
                        label="Audio clip (up to 30 s)"),
        outputs=[gr.Label(num_top_classes=3, label="Top 3 genres"),
                 gr.Plot(label="Probabilities"), gr.Plot(label="Spectrogram")],
        title="Music genre recognition",
        description=(f"Model: {pred.name}. Trained on FMA-medium ({len(pred.genres)} genres).\n\n"
                     "Microphone: the browser must allow it for this page. If it says 'No microphone found', "
                     "click the icon left of the address bar, set Microphone to Allow, and reload."),
        flagging_mode="never",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    run_dir = Path(args.run) if args.run else pick_run()
    run_dir = run_dir if run_dir.is_absolute() else ROOT / run_dir
    pred = Predictor(run_dir)
    print(f"serving {pred.name}")
    print(f"Open http://127.0.0.1:{args.port} (use this exact address, not 'localhost'). "
          "Keep this window open; press Ctrl+C to stop.", flush=True)
    make_ui(pred).launch(server_name="127.0.0.1", server_port=args.port, inbrowser=True)


if __name__ == "__main__":
    main()
