"""Music Genre Classifier web demo: upload or record a clip, see the top-3 genres and the mel-spectrogram.

The page is app/web/index.html (plain HTML/JS); this file serves it with FastAPI and runs the model.
The previous Gradio version is kept as app/demo_gradio.py.

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

from fastapi import FastAPI, File, HTTPException, UploadFile  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402

# Heavy libraries (torch, transformers, ...) are imported in a background thread (see load_heavy) so the web
# page opens immediately and shows "loading" instead of the terminal sitting silent for minutes.
np = torch = audio = build_model = DEVICE = None


def load_heavy() -> None:
    global np, torch, audio, build_model, DEVICE
    import numpy as _np
    import torch as _torch

    import audio as _audio
    from models import build_model as _build

    np, torch, audio, build_model = _np, _torch, _audio, _build
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    def __call__(self, path: str):
        with torch.no_grad():
            return self._predict(path)

    def _predict(self, path: str):
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


WEB = ROOT / "app" / "web"
MAX_UPLOAD = 60 * 1024 * 1024


def spec_image(mel: np.ndarray, max_cols: int = 400) -> dict:
    """Log-mel -> small uint8 grid (rows = mel bins, low to high) the page draws on a canvas."""
    m = mel.astype(np.float32)
    n = max(1, int(np.ceil(m.shape[1] / max_cols)))
    cols = m.shape[1] // n * n
    m = m[:, :cols].reshape(m.shape[0], cols // n, n).mean(-1)
    lo, hi = np.percentile(m, [2, 99.5])
    img = np.clip((m - lo) / max(hi - lo, 1e-6), 0, 1)
    img = (img * 255).astype(np.uint8)
    import base64
    return {"w": int(img.shape[1]), "h": int(img.shape[0]), "data": base64.b64encode(img.tobytes()).decode()}


def make_app(state: dict):
    import tempfile
    import threading
    import traceback

    app = FastAPI(title="Music Genre Classifier")
    lock = threading.Lock()  # one prediction at a time on the GPU

    @app.get("/")
    def index():
        return FileResponse(WEB / "index.html")

    @app.get("/api/info")
    def info():
        pred = state.get("pred")
        if state.get("error"):
            return {"ready": False, "error": state["error"]}
        if pred is None:
            return {"ready": False}
        return {"ready": True, "model": pred.cfg["model"]["name"], "run": pred.name, "n_genres": len(pred.genres)}

    @app.post("/api/predict")
    def predict(file: UploadFile = File(...)):
        pred = state.get("pred")
        if pred is None:
            raise HTTPException(503, "The model is still loading. Wait a moment and try again.")
        suffix = Path(file.filename or "audio").suffix or ".bin"
        tmp = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
                tmp = f.name
                data = file.file.read(MAX_UPLOAD + 1)
                if len(data) > MAX_UPLOAD:
                    raise HTTPException(413, "File is too large (limit 60 MB).")
                f.write(data)
            with lock:
                probs, mel = pred(tmp)
        except HTTPException:
            raise
        except Exception as e:  # noqa: BLE001
            tb = traceback.format_exc()
            print(tb, flush=True)
            try:
                (ROOT / "logs").mkdir(exist_ok=True)
                with open(ROOT / "logs" / "demo_errors.log", "a", encoding="utf-8") as lf:
                    lf.write(f"--- {file.filename}\n{tb}\n")
            except OSError:
                pass
            raise HTTPException(400, f"{type(e).__name__}: {e}")
        finally:
            if tmp:
                try:
                    Path(tmp).unlink()
                except OSError:
                    pass
        top = np.argsort(-probs)[:3]
        return {"top": [{"genre": pred.genres[i], "prob": float(probs[i])} for i in top], "spec": spec_image(mel)}

    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    run_dir = Path(args.run) if args.run else pick_run()
    run_dir = run_dir if run_dir.is_absolute() else ROOT / run_dir

    import threading
    state: dict = {}

    def loader() -> None:
        try:
            load_heavy()
            state["pred"] = Predictor(run_dir)
            print(f"Model ready: {state['pred'].name}. You can upload or record now.", flush=True)
        except BaseException as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            state["error"] = f"{type(e).__name__}: {e}"

    threading.Thread(target=loader, daemon=True).start()
    url = f"http://127.0.0.1:{args.port}"
    print(f"Music Genre Classifier is opening at {url}", flush=True)
    print("The page opens right away. The model loads in the background (1-3 minutes the first time after a "
          "restart); the page tells you when it is ready. Keep this window open; press Ctrl+C to stop.", flush=True)
    if not args.no_browser:
        import webbrowser
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    import uvicorn
    uvicorn.run(make_app(state), host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
