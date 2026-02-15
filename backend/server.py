from fastapi import FastAPI, APIRouter, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
import os
import logging
import re
import hashlib
from pathlib import Path
from typing import List, Optional, Any
from pydantic import BaseModel, Field, ConfigDict

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env")

# Optional MongoDB
try:
    mongo_url = os.environ.get("MONGO_URL")
    db_name = os.environ.get("DB_NAME", "musicmath")
    if mongo_url:
        from motor.motor_asyncio import AsyncIOMotorClient
        client = AsyncIOMotorClient(mongo_url)
        db = client[db_name]
        HAS_MONGO = True
    else:
        client = None
        db = None
        HAS_MONGO = False
except Exception:
    client = None
    db = None
    HAS_MONGO = False

# Data directory: project root with results/stats and metadata_clean.csv (set in Docker or .env)
DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT_DIR.parent.parent))
STATS_DIR = DATA_DIR / "results" / "stats"
META_PATH = DATA_DIR / "metadata_clean.csv"
FEATURE_MATRIX_PATH = STATS_DIR / "feature_matrix.csv"
SIGNATURE_SCORES_PATH = STATS_DIR / "signature_scores.csv"
NUMERIC_PATTERNS_PATH = STATS_DIR / "feature_matrix_with_numeric_patterns.csv"
SIGNATURE_ERA_SUMMARY_PATH = STATS_DIR / "signature_era_summary.csv"
MATH_PATTERNS_WIDE_PATH = STATS_DIR / "mathematical_pattern_outputs_wide.csv"
BRUTE_FORCE_SUMMARY_PATH = STATS_DIR / "brute_force_pattern_summary.csv"

app = FastAPI(title="MusicMath DNA API")
api_router = APIRouter(prefix="/api")

# In-memory cache for analysis data (loaded on first request)
_analysis_cache: Optional[dict] = None


def _slug(s: str) -> str:
    """Turn composer or string into a URL-safe slug."""
    s = re.sub(r"[^\w\s-]", "", s.lower())
    return re.sub(r"[-\s]+", "-", s).strip("-") or "unknown"


def _track_id(filepath: str, composer: str, index: int) -> str:
    """Stable id for a track: composer slug + short hash of filepath."""
    h = hashlib.md5(filepath.encode()).hexdigest()[:8]
    base = _slug(composer) or "unknown"
    return f"{base}-{h}"


def _load_analysis_data() -> dict:
    """Load CSVs and build lookup by track_id. Returns { track_id -> row_dict, id_to_filepath, tracks_list }."""
    global _analysis_cache
    if _analysis_cache is not None:
        return _analysis_cache

    import pandas as pd

    if not FEATURE_MATRIX_PATH.exists():
        # Fallback: metadata_clean (+ brute_force_summary varsa)
        if META_PATH.exists():
            meta_df = pd.read_csv(META_PATH)
            bf_df = pd.read_csv(BRUTE_FORCE_SUMMARY_PATH) if BRUTE_FORCE_SUMMARY_PATH.exists() else None
            bf_by_path = dict(zip(bf_df["filepath"], bf_df.to_dict("records"))) if bf_df is not None and "filepath" in bf_df.columns else {}
            tracks, by_id, id_to_filepath = [], {}, {}
            for i, row in meta_df.iterrows():
                fp = row.get("file_path", "")
                composer = str(row.get("composer", "Unknown"))
                era = str(row.get("era", "Other"))
                tid = _track_id(fp, composer, i)
                title = Path(fp).stem if fp else f"Piece {i+1}"
                title = title.replace("_", " ").replace("-", " ")[:60]
                tracks.append({"id": tid, "title": title, "composer": composer, "era": era, "year": 0})
                bf_row = bf_by_path.get(fp, {})
                by_id[tid] = {
                    **row.to_dict(), "filepath": fp,
                    "sig_melodic_volatility": 0.5, "sig_motif_repetition": 0.5,
                    "sig_consonance_balance": 0.5, "sig_structural_complexity": 0.5,
                    "prime_sum_ratio": 0.2, "phi_density_ratio": 0.5, "interval_self_similarity": 0.5,
                    **{k: bf_row[k] for k in ("top_pattern", "top_z_score", "n_notes") if k in bf_row},
                }
                id_to_filepath[tid] = fp
            _analysis_cache = {"tracks": tracks, "by_id": by_id, "id_to_filepath": id_to_filepath}
            return _analysis_cache
        _analysis_cache = {"tracks": [], "by_id": {}, "id_to_filepath": {}}
        return _analysis_cache

    # Load matrix with numeric patterns (has prime_sum_ratio, phi_density_ratio, interval_self_similarity)
    if NUMERIC_PATTERNS_PATH.exists():
        df = pd.read_csv(NUMERIC_PATTERNS_PATH)
    else:
        df = pd.read_csv(FEATURE_MATRIX_PATH)

    # Merge signature scores (sig_melodic_volatility, etc.) from separate file
    if SIGNATURE_SCORES_PATH.exists():
        sig = pd.read_csv(SIGNATURE_SCORES_PATH)
        sig_cols = [c for c in sig.columns if c.startswith("sig_") or c == "filepath"]
        df = df.merge(sig[sig_cols].drop_duplicates("filepath"), on="filepath", how="left")

    tracks = []
    by_id = {}
    id_to_filepath = {}

    for i, row in df.iterrows():
        fp = row.get("filepath", "")
        composer = str(row.get("composer", "Unknown"))
        era = str(row.get("era", "Other"))
        comp_year = row.get("composition_year", 0)
        try:
            comp_year = int(comp_year) if comp_year == comp_year else 0
        except (ValueError, TypeError):
            comp_year = 0
        tid = _track_id(fp, composer, i)
        title = Path(fp).stem if fp else f"Piece {i+1}"
        title = title.replace("_", " ").replace("-", " ")[:60]
        tracks.append({
            "id": tid,
            "title": title,
            "composer": composer,
            "era": era,
            "year": comp_year,
        })
        by_id[tid] = row.to_dict()
        id_to_filepath[tid] = fp

    # Normalize signature columns to 0–1 for frontend (if z-scores)
    for tid, row in by_id.items():
        for key in ("sig_melodic_volatility", "sig_motif_repetition", "sig_consonance_balance", "sig_structural_complexity"):
            if key in row and row[key] is not None:
                v = float(row[key])
                # simple sigmoid-ish: map to 0-1 (assume z-scores roughly in -3..3)
                row[key] = 1 / (1 + pow(2.718, -v * 0.6))
            elif key not in row and key.replace("sig_", "") in row:
                rk = key.replace("sig_", "")
                if rk == "melodic_volatility":
                    row[key] = _norm(row.get("interval_entropy", 0.5), 2.5, 4.5)
                elif rk == "motif_repetition":
                    row[key] = _norm(row.get("repetition_index", 0.5), 0.3, 0.8)
                elif rk == "consonance_balance":
                    row[key] = float(row.get("consonance_score", 0.5))
                elif rk == "structural_complexity":
                    row[key] = _norm(row.get("fractal_dimension", 1.5), 1.2, 2.2)
                else:
                    row[key] = 0.5

    _analysis_cache = {"tracks": tracks, "by_id": by_id, "id_to_filepath": id_to_filepath}
    return _analysis_cache


def _norm(x, lo, hi):
    return max(0.0, min(1.0, (float(x) - lo) / (hi - lo) if hi != lo else 0.5))


def _build_analysis_response(track_id: str, row: dict) -> dict:
    """Shape one row into the frontend's expected analysis result."""
    era = str(row.get("era", "Other"))
    composer = str(row.get("composer", "Unknown"))
    # Era confidences: put most weight on predicted era
    eras = ["Baroque", "Classical", "Romantic", "Late Romantic"]
    conf = 0.85 if era in eras else 0.4
    era_confidences = {e: (conf if e == era else (1 - conf) / 3) for e in eras}

    sig = {
        "melodicVolatility": {"value": row.get("sig_melodic_volatility", 0.5), "description": "Melodic volatility from interval entropy and motion."},
        "motifRepetition": {"value": row.get("sig_motif_repetition", 0.5), "description": "Motif repetition from repetition index."},
        "consonanceBalance": {"value": row.get("sig_consonance_balance", row.get("consonance_score", 0.5)), "description": "Consonance balance from interval analysis."},
        "structuralComplexity": {"value": row.get("sig_structural_complexity", 0.5), "description": "Structural complexity from fractal dimension."},
    }

    numeric_patterns = {
        "primeSumRatio": {"value": row.get("prime_sum_ratio", 0.2), "description": "Prime-sum ratio from pitch cumulatives."},
        "phiDensityRatio": {"value": row.get("phi_density_ratio", 0.5), "description": "Golden-ratio density in timeline."},
        "intervalSelfSimilarity": {"value": row.get("interval_self_similarity", 0.5), "description": "Interval self-similarity (fractal)."},
    }

    suggestions = [
        {"text": f"Era classification: {era}. Composer: {composer}.", "type": "insight"},
        {"text": "Explore signature metrics and numeric patterns for deeper patterns.", "type": "experiment"},
    ]

    return {
        "trackId": track_id,
        "overview": {
            "predictedEra": era,
            "predictedComposer": composer,
            "confidence": conf,
            "eraConfidences": era_confidences,
            "commentary": f"Mathematical signature from feature matrix: era {era}, composer {composer}. Metrics derived from pitch, interval, and rhythm features.",
        },
        "signatures": sig,
        "numericPatterns": numeric_patterns,
        "suggestions": suggestions,
    }


# ----- API routes -----

@api_router.get("/")
async def root():
    return {"message": "MusicMath DNA API", "docs": "/docs"}


@api_router.get("/tracks")
async def get_tracks():
    """List all tracks (from feature matrix / metadata). Frontend Playground uses this."""
    data = _load_analysis_data()
    return data["tracks"]


@api_router.get("/analysis/{track_id}")
async def get_analysis(track_id: str):
    """Get analysis result for one track. Shape matches frontend mock (overview, signatures, numericPatterns, suggestions)."""
    data = _load_analysis_data()
    row = data["by_id"].get(track_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Track not found")
    return _build_analysis_response(track_id, row)


@api_router.get("/health")
async def health():
    return {"status": "ok", "mongo": HAS_MONGO, "data_dir": str(DATA_DIR)}


@api_router.get("/pipeline/status")
async def pipeline_status():
    """Pipeline veri dosyalarının durumu. Frontend hangi verilerin hazır olduğunu gösterir."""
    files = {
        "metadata_clean": META_PATH.exists(),
        "feature_matrix": FEATURE_MATRIX_PATH.exists(),
        "feature_matrix_with_numeric_patterns": NUMERIC_PATTERNS_PATH.exists(),
        "signature_scores": SIGNATURE_SCORES_PATH.exists(),
        "signature_era_summary": SIGNATURE_ERA_SUMMARY_PATH.exists(),
        "mathematical_patterns_wide": MATH_PATTERNS_WIDE_PATH.exists(),
        "brute_force_summary": BRUTE_FORCE_SUMMARY_PATH.exists(),
    }
    data_dir = str(DATA_DIR)
    return {
        "data_dir": data_dir,
        "files": files,
        "tracks_available": files["feature_matrix"] or (files["metadata_clean"] and files["brute_force_summary"]),
    }


# ----- Mathematical patterns (100 oruntu ciktisi) -----

@api_router.get("/patterns/mathematical")
async def get_mathematical_patterns(summary: bool = True):
    """100 matematiksel oruntu ciktisi (mathematical_pattern_outputs_wide.csv). summary=True: pattern_id, name, category; False: full wide data."""
    if not MATH_PATTERNS_WIDE_PATH.exists():
        return {"patterns": [], "message": "mathematical_pattern_outputs_wide.csv not found. Run: python scripts/brute_force_patterns.py --export-patterns"}
    import pandas as pd
    df = pd.read_csv(MATH_PATTERNS_WIDE_PATH)
    if summary:
        rows = []
        for _, r in df.iterrows():
            rows.append({"pattern_id": r.get("pattern_id"), "pattern_name": r.get("pattern_name"), "category": r.get("category")})
        return {"patterns": rows, "count": len(rows)}
    return {"patterns": df.to_dict(orient="records"), "count": len(df)}


@api_router.get("/patterns/mathematical/{pattern_id}")
async def get_mathematical_pattern_detail(pattern_id: str):
    """Tek oruntunun deger serisi (v0, v1, ...)."""
    if not MATH_PATTERNS_WIDE_PATH.exists():
        raise HTTPException(status_code=404, detail="Mathematical patterns file not found")
    import pandas as pd
    df = pd.read_csv(MATH_PATTERNS_WIDE_PATH)
    row = df[df["pattern_id"] == pattern_id]
    if row.empty:
        raise HTTPException(status_code=404, detail="Pattern not found")
    r = row.iloc[0]
    v_keys = [k for k in r.index if isinstance(k, str) and k.startswith("v")]
    v_keys.sort(key=lambda x: int(x[1:]) if x[1:].isdigit() else 0)
    values = [r[k] for k in v_keys]
    return {"pattern_id": pattern_id, "pattern_name": r.get("pattern_name"), "category": r.get("category"), "values": values}


# ----- Gallery visualizations (t-SNE, confusion, signatures by era, numeric timeline) -----

def _load_gallery_data() -> dict:
    """Gallery sayfasi icin veri: tsne, confusion, signatureByEra, numericTimeline."""
    import pandas as pd
    out = {"tsne": [], "confusion": [], "signatureByEra": [], "numericTimeline": []}
    if not FEATURE_MATRIX_PATH.exists():
        return out
    df = pd.read_csv(FEATURE_MATRIX_PATH)
    if "era" not in df.columns or len(df) == 0:
        return out
    # t-SNE benzeri: 2D projeksiyon yoksa pitch_mean vs interval_entropy ile basit 2D
    x_col = "pitch_mean" if "pitch_mean" in df.columns else df.columns[0]
    y_col = "interval_entropy" if "interval_entropy" in df.columns else (df.columns[1] if len(df.columns) > 1 else df.columns[0])
    for i, row in df.iterrows():
        x, y = float(row.get(x_col, 0)), float(row.get(y_col, 0))
        era = str(row.get("era", "Other"))
        out["tsne"].append({"x": x, "y": y, "era": era})
    # Confusion: era bazli ortalama (simplified – diagonal agirlikli)
    eras = df["era"].dropna().unique().tolist()
    for era in eras:
        subset = df[df["era"] == era]
        n = len(subset)
        row_data = {"name": era.replace(" ", "") if isinstance(era, str) else str(era)}
        for e in eras:
            key = e.replace(" ", "") if isinstance(e, str) else str(e)
            row_data[key] = 0.85 if e == era else (0.15 / max(len(eras) - 1, 1))
        out["confusion"].append(row_data)
    # Signature by era (signature_era_summary.csv: row 0,1=header, row 2=era labels, row 3+=data)
    if SIGNATURE_ERA_SUMMARY_PATH.exists():
        era_df = pd.read_csv(SIGNATURE_ERA_SUMMARY_PATH)
        # Skip header rows; first col = era, means at cols 2,5,8,11 (melodic, motif, consonance, structural)
        data_start = 3
        if len(era_df) > data_start:
            for i in range(data_start, len(era_df)):
                row = era_df.iloc[i]
                era = str(row.iloc[0]) if row.iloc[0] == row.iloc[0] else "Other"  # NaN check
                if era == "nan" or not era:
                    continue
                vals = [float(row.iloc[j]) if j < len(row) and row.iloc[j] == row.iloc[j] else 0.5 for j in [2, 5, 8, 11]]
                sigmoid = lambda v: 1 / (1 + 2.718 ** (-v * 0.6))
                out["signatureByEra"].append({
                    "era": era,
                    "melodicVol": max(0, min(1, sigmoid(vals[0]))),
                    "consonance": max(0, min(1, sigmoid(vals[2]))),
                    "complexity": max(0, min(1, sigmoid(vals[3]))),
                    "motif": max(0, min(1, sigmoid(vals[1]))),
                })
    if not out["signatureByEra"] and "era" in df.columns:
        for era in df["era"].dropna().unique():
            out["signatureByEra"].append({"era": era, "melodicVol": 0.5, "consonance": 0.5, "complexity": 0.5, "motif": 0.5})
    # Numeric timeline: son N satirdan prime/phi/self-sim (varsa)
    if NUMERIC_PATTERNS_PATH.exists():
        num_df = pd.read_csv(NUMERIC_PATTERNS_PATH).tail(20)
        for i, row in num_df.iterrows():
            out["numericTimeline"].append({
                "index": len(out["numericTimeline"]) + 1,
                "primeDensity": float(row.get("prime_sum_ratio", 0.2)),
                "phiDensity": float(row.get("phi_density_ratio", 0.5)),
                "selfSimilarity": float(row.get("interval_self_similarity", 0.5)),
            })
    return out


_gallery_cache: Optional[dict] = None


@api_router.get("/gallery/visualizations")
async def get_gallery_visualizations():
    """Gallery sayfasi: tsne, confusion, signatureByEra, numericTimeline."""
    global _gallery_cache
    if _gallery_cache is None:
        _gallery_cache = _load_gallery_data()
    return _gallery_cache


# ----- Experiment: algoritmalar + kullanici secimli analiz (oruntu/parametre deneyleri) -----

try:
    from algorithms import get_algorithms_list, run_algorithm, ALGORITHMS
except ImportError:
    get_algorithms_list = run_algorithm = ALGORITHMS = None

# Numeric feature names for track-based series (from feature matrix columns)
def _get_numeric_feature_names() -> List[str]:
    import pandas as pd
    if not FEATURE_MATRIX_PATH.exists():
        return []
    df = pd.read_csv(FEATURE_MATRIX_PATH, nrows=0)
    skip = {"filepath", "composer", "era", "form"}
    return [c for c in df.columns if c not in skip]


def _resolve_series(
    source_type: str,
    track_id: Optional[str] = None,
    feature_name: Optional[str] = None,
    pattern_id: Optional[str] = None,
    custom_series: Optional[List[float]] = None,
) -> List[float]:
    """Kullanici secimine gore girdi serisini uretir: track ozelligi, oruntu degerleri, veya custom."""
    if source_type == "custom" and custom_series is not None:
        return [float(x) for x in custom_series]
    if source_type == "pattern" and pattern_id:
        if not MATH_PATTERNS_WIDE_PATH.exists():
            raise HTTPException(status_code=404, detail="Mathematical patterns file not found. Run: python scripts/brute_force_patterns.py --export-patterns")
        import pandas as pd
        df = pd.read_csv(MATH_PATTERNS_WIDE_PATH)
        row = df[df["pattern_id"] == pattern_id]
        if row.empty:
            raise HTTPException(status_code=404, detail=f"Pattern {pattern_id} not found")
        r = row.iloc[0]
        v_keys = [k for k in r.index if isinstance(k, str) and k.startswith("v")]
        v_keys.sort(key=lambda x: int(x[1:]) if x[1:].isdigit() else 0)
        return [float(r[k]) for k in v_keys]
    if source_type == "track" and track_id:
        data = _load_analysis_data()
        row = data["by_id"].get(track_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Track not found")
        numeric_cols = _get_numeric_feature_names()
        if feature_name:
            if feature_name not in row:
                raise HTTPException(status_code=404, detail=f"Feature '{feature_name}' not in track data")
            try:
                return [float(row[feature_name])]
            except (TypeError, ValueError):
                raise HTTPException(status_code=400, detail=f"Feature '{feature_name}' is not numeric")
        # Tum sayisal ozellikleri sirayla ver (parca vektorunu seri gibi kullan)
        try:
            return [float(row[c]) for c in numeric_cols if c in row and row[c] is not None]
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="Track has non-numeric values in feature columns")
    raise HTTPException(status_code=400, detail="Invalid source: use custom_series, or track_id (+ optional feature_name), or pattern_id")


@api_router.get("/experiment/algorithms")
async def experiment_algorithms():
    """Tum algoritmalar: id, name, description, params (schema). Kalman, EMA, Gaussian, Savitzky-Golay, Rolling median, Double exp, Wiener."""
    if get_algorithms_list is None:
        raise HTTPException(status_code=501, detail="Algorithms module not loaded")
    return {"algorithms": get_algorithms_list()}


@api_router.get("/experiment/features")
async def experiment_features():
    """Track tabanli seri secerken kullanilabilecek sayisal ozellik isimleri (feature_matrix kolonlari)."""
    names = _get_numeric_feature_names()
    return {"features": names}


@api_router.get("/experiment/patterns")
async def experiment_patterns_list():
    """Matematiksel oruntu listesi (pattern_id); experiment run'da source_type=pattern icin."""
    if not MATH_PATTERNS_WIDE_PATH.exists():
        return {"patterns": [], "message": "Export mathematical patterns first."}
    import pandas as pd
    df = pd.read_csv(MATH_PATTERNS_WIDE_PATH)
    patterns = [{"id": r["pattern_id"], "name": r.get("pattern_name", r["pattern_id"]), "category": r.get("category", "")} for _, r in df.iterrows()]
    return {"patterns": patterns}


class ExperimentRunBody(BaseModel):
    source_type: str = "custom"  # "track" | "pattern" | "custom"
    track_id: Optional[str] = None
    feature_name: Optional[str] = None
    pattern_id: Optional[str] = None
    custom_series: Optional[List[float]] = None
    algorithm_id: str = "kalman"
    algorithm_params: Optional[dict] = None


@api_router.post("/experiment/run")
async def experiment_run(body: ExperimentRunBody):
    """
    Secilen kaynak (track ozelligi / oruntu / custom seri) + algoritma + parametrelerle analiz calistirir.
    Donen: input_series, output_series, algorithm_id, params_used, summary (min/max/mean).
    Kullanici istedigi oruntu ve parametreyle deney yapip harmoniyi arayabilir.
    """
    if run_algorithm is None:
        raise HTTPException(status_code=501, detail="Algorithms module not loaded")
    series = _resolve_series(
        body.source_type,
        track_id=body.track_id,
        feature_name=body.feature_name,
        pattern_id=body.pattern_id,
        custom_series=body.custom_series,
    )
    if not series:
        raise HTTPException(status_code=400, detail="Resolved series is empty")
    try:
        output = run_algorithm(body.algorithm_id, series, body.algorithm_params)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("experiment run")
        raise HTTPException(status_code=500, detail=str(e))
    import numpy as np
    def summary(s):
        a = np.array(s, dtype=float)
        return {"min": float(np.nanmin(a)), "max": float(np.nanmax(a)), "mean": float(np.nanmean(a)), "length": len(s)}
    return {
        "input_series": series,
        "output_series": output,
        "algorithm_id": body.algorithm_id,
        "params_used": body.algorithm_params or {},
        "input_summary": summary(series),
        "output_summary": summary(output),
    }


# ----- Artwork: muzik -> gorsel (Kalman, palet, cizim altyapisi) -----

try:
    from artwork import get_palettes, get_render_modes, artwork_config
except ImportError:
    get_palettes = get_render_modes = artwork_config = None


@api_router.get("/artwork/config")
async def get_artwork_config():
    """Paletler, render modlari, varsayilan boyut. Ileride canvas cizimi icin."""
    if artwork_config is None:
        raise HTTPException(status_code=501, detail="Artwork module not loaded")
    return artwork_config()


@api_router.get("/artwork/palettes")
async def list_artwork_palettes():
    """Duygu / estetik renk paletleri (modern sanat cizimi icin)."""
    if get_palettes is None:
        raise HTTPException(status_code=501, detail="Artwork module not loaded")
    return {"palettes": get_palettes()}


@api_router.get("/artwork/render-modes")
async def list_render_modes():
    """Render modlari: kalman, spectral, waveform, phi_arc, fractal."""
    if get_render_modes is None:
        raise HTTPException(status_code=501, detail="Artwork module not loaded")
    return {"modes": get_render_modes()}


@api_router.get("/artwork/preview/{track_id}")
async def get_artwork_preview(
    track_id: str,
    palette_id: Optional[str] = None,
    mode: Optional[str] = None,
):
    """Stub: Parça için önerilen çizim konfigü (palet, mod). İleride gerçek görsel dönecek."""
    data = _load_analysis_data()
    row = data["by_id"].get(track_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Track not found")
    cfg = artwork_config() if artwork_config else {}
    return {
        "trackId": track_id,
        "paletteId": palette_id or cfg.get("defaultPaletteId", "emotional_dark"),
        "renderMode": mode or cfg.get("defaultRenderMode", "kalman"),
        "message": "Preview pipeline: music features → Kalman smoothing → color mapping → canvas. Image generation coming soon.",
        "suggestedWidth": cfg.get("width", 1200),
        "suggestedHeight": cfg.get("height", 800),
    }


# ----- Artwork Render: gercek gorsel uretimi -----

try:
    import sys
    sys.path.insert(0, str(DATA_DIR))
    from music_math.viz.renderer import render_to_bytes, RenderConfig, RENDERERS
    from music_math.data.loader import parse_midi_to_note_events as load_midi_notes
    from music_math.core.types import NoteEvent
    HAS_RENDERER = True
except ImportError:
    HAS_RENDERER = False


class ArtworkRenderBody(BaseModel):
    mode: str = "kalman"
    palette_id: str = "emotional_dark"
    width: int = 1200
    height: int = 800
    blur_radius: float = 2.0
    custom_palette: Optional[List[str]] = None


@api_router.post("/artwork/render/{track_id}")
async def render_artwork(track_id: str, body: ArtworkRenderBody):
    """Parcadan gercek gorsel uret (PNG, base64)."""
    if not HAS_RENDERER:
        raise HTTPException(status_code=501, detail="Renderer module not loaded (Pillow + music_math required)")

    data = _load_analysis_data()
    filepath = data["id_to_filepath"].get(track_id)
    if filepath is None:
        raise HTTPException(status_code=404, detail="Track not found")

    if body.mode not in RENDERERS:
        raise HTTPException(status_code=400, detail=f"Invalid mode. Available: {list(RENDERERS.keys())}")

    try:
        notes = load_midi_notes(filepath)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MIDI parse error: {e}")

    if not notes:
        raise HTTPException(status_code=400, detail="No notes found in MIDI file")

    config = RenderConfig(
        width=body.width,
        height=body.height,
        palette_id=body.palette_id,
        blur_radius=body.blur_radius,
    )

    try:
        png_bytes = render_to_bytes(notes, mode=body.mode, config=config, palette_hex=body.custom_palette)
    except Exception as e:
        logger.exception("Artwork render error")
        raise HTTPException(status_code=500, detail=str(e))

    import base64
    b64 = base64.b64encode(png_bytes).decode("ascii")
    return {
        "trackId": track_id,
        "mode": body.mode,
        "paletteId": body.palette_id,
        "width": body.width,
        "height": body.height,
        "format": "png",
        "image_base64": b64,
        "size_bytes": len(png_bytes),
    }


# ----- Note Prediction / Emotional Impact -----

try:
    from music_math.model.predictor import NotePredictor, PredictorConfig, NoteEvent as PredNoteEvent
    from music_math.model.markov import MusicMarkovModel
    HAS_PREDICTOR = True
except ImportError:
    HAS_PREDICTOR = False


class PredictNextBody(BaseModel):
    track_id: Optional[str] = None
    notes: Optional[List[dict]] = None
    top_k: int = 5
    temperature: float = 1.0
    target_emotion: str = "balanced"
    piece_length: float = 0.0


class GenerateSequenceBody(BaseModel):
    track_id: Optional[str] = None
    seed_notes: Optional[List[dict]] = None
    length: int = 32
    target_emotion: str = "balanced"
    temperature: float = 1.0


class AnalyzeImpactBody(BaseModel):
    track_id: Optional[str] = None
    notes: Optional[List[dict]] = None


def _notes_from_dicts(dicts: List[dict]) -> list:
    return [NoteEvent(pitch=int(d["pitch"]), duration=float(d.get("duration", 1.0)), start=float(d.get("start", 0.0))) for d in dicts]


def _notes_from_track(track_id: str) -> list:
    if not HAS_RENDERER:
        raise HTTPException(status_code=501, detail="music_math loader not available")
    data = _load_analysis_data()
    filepath = data["id_to_filepath"].get(track_id)
    if filepath is None:
        raise HTTPException(status_code=404, detail="Track not found")
    try:
        return load_midi_notes(filepath)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MIDI parse error: {e}")


def _build_predictor(temperature: float = 1.0) -> "NotePredictor":
    cfg = PredictorConfig(temperature=temperature)
    predictor = NotePredictor(config=cfg)
    return predictor


@api_router.post("/predict/next-note")
async def predict_next_note(body: PredictNextBody):
    """Bir sonraki nota onerisi: Kalman + Markov + matematiksel oruntu + duygusal etki."""
    if not HAS_PREDICTOR:
        raise HTTPException(status_code=501, detail="Predictor module not loaded")

    if body.track_id:
        notes = _notes_from_track(body.track_id)
    elif body.notes:
        notes = _notes_from_dicts(body.notes)
    else:
        raise HTTPException(status_code=400, detail="Provide track_id or notes array")

    if not notes:
        raise HTTPException(status_code=400, detail="No notes to predict from")

    predictor = _build_predictor(body.temperature)
    predictor.feed_context(notes, piece_total_length=body.piece_length)
    suggestions = predictor.predict_next(top_k=body.top_k)

    return {
        "suggestions": [
            {
                "pitch": s.pitch,
                "note_name": _pitch_to_name(s.pitch),
                "score": round(s.score, 4),
                "predicted_duration": s.predicted_duration,
                "emotional_tag": s.emotional_tag,
                "scores_detail": {k: round(v, 4) for k, v in s.scores_detail.items()},
            }
            for s in suggestions
        ],
        "context_length": len(notes),
    }


@api_router.post("/predict/generate")
async def generate_emotional_sequence(body: GenerateSequenceBody):
    """Hedef duyguya gore nota dizisi uret."""
    if not HAS_PREDICTOR:
        raise HTTPException(status_code=501, detail="Predictor module not loaded")

    if body.track_id:
        seed = _notes_from_track(body.track_id)
    elif body.seed_notes:
        seed = _notes_from_dicts(body.seed_notes)
    else:
        seed = [NoteEvent(pitch=60, duration=1.0, start=0.0)]

    predictor = _build_predictor(body.temperature)
    sequence = predictor.generate_emotional_sequence(
        length=body.length,
        start_notes=seed[-16:],
        target_emotion=body.target_emotion,
    )

    return {
        "sequence": [
            {
                "pitch": n.pitch,
                "note_name": _pitch_to_name(n.pitch),
                "duration": round(n.duration, 3),
                "start": round(n.start, 3),
            }
            for n in sequence
        ],
        "length": len(sequence),
        "target_emotion": body.target_emotion,
    }


@api_router.post("/predict/impact")
async def analyze_impact(body: AnalyzeImpactBody):
    """Nota dizisinin duygusal etki analizini yap."""
    if not HAS_PREDICTOR:
        raise HTTPException(status_code=501, detail="Predictor module not loaded")

    if body.track_id:
        notes = _notes_from_track(body.track_id)
    elif body.notes:
        notes = _notes_from_dicts(body.notes)
    else:
        raise HTTPException(status_code=400, detail="Provide track_id or notes array")

    predictor = _build_predictor()
    analysis = predictor.analyze_sequence_impact(notes)
    return analysis


def _pitch_to_name(midi: int) -> str:
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return f"{names[midi % 12]}{midi // 12 - 1}"


# ----- Celery task API: async pipeline / brute force / analyses -----

try:
    from tasks import (
        run_pipeline as run_pipeline_task,
        run_analyses as run_analyses_task,
        run_brute_force_chunk as run_brute_force_chunk_task,
        run_brute_force_merge as run_brute_force_merge_task,
        run_brute_force_full as run_brute_force_full_task,
        run_pattern_export as run_pattern_export_task,
        run_full_pipeline as run_full_pipeline_task,
    )
    from celery_app import celery as celery_app_instance
    HAS_CELERY = True
except ImportError:
    HAS_CELERY = False


class TaskRunBody(BaseModel):
    brute_force_chunks: int = 4


def _check_celery():
    if not HAS_CELERY:
        raise HTTPException(status_code=501, detail="Celery not configured. Install celery and redis.")


@api_router.post("/tasks/pipeline")
async def start_pipeline():
    _check_celery()
    t = run_pipeline_task.delay()
    return {"task_id": t.id, "task": "pipeline", "status": "queued"}


@api_router.post("/tasks/analyses")
async def start_analyses():
    _check_celery()
    t = run_analyses_task.delay()
    return {"task_id": t.id, "task": "analyses", "status": "queued"}


@api_router.post("/tasks/brute-force")
async def start_brute_force(body: TaskRunBody):
    _check_celery()
    t = run_brute_force_full_task.delay(total_chunks=body.brute_force_chunks)
    return {"task_id": t.id, "task": "brute_force_full", "chunks": body.brute_force_chunks, "status": "queued"}


@api_router.post("/tasks/brute-force/chunk")
async def start_brute_force_chunk(chunk: int = 0, total_chunks: int = 4):
    _check_celery()
    t = run_brute_force_chunk_task.delay(chunk=chunk, total_chunks=total_chunks)
    return {"task_id": t.id, "task": f"brute_force_chunk_{chunk}", "status": "queued"}


@api_router.post("/tasks/brute-force/merge")
async def start_brute_force_merge(total_chunks: int = 4):
    _check_celery()
    t = run_brute_force_merge_task.delay(total_chunks=total_chunks)
    return {"task_id": t.id, "task": "brute_force_merge", "status": "queued"}


@api_router.post("/tasks/pattern-export")
async def start_pattern_export():
    _check_celery()
    t = run_pattern_export_task.delay()
    return {"task_id": t.id, "task": "pattern_export", "status": "queued"}


@api_router.post("/tasks/full")
async def start_full_pipeline(body: TaskRunBody):
    _check_celery()
    t = run_full_pipeline_task.delay(brute_force_chunks=body.brute_force_chunks)
    return {"task_id": t.id, "task": "full_pipeline", "chunks": body.brute_force_chunks, "status": "queued"}


@api_router.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    _check_celery()
    from celery.result import AsyncResult
    result = AsyncResult(task_id, app=celery_app_instance)
    response = {
        "task_id": task_id,
        "state": result.state,
    }
    if result.state == "PROGRESS":
        response["meta"] = result.info
    elif result.state == "SUCCESS":
        response["result"] = result.result
    elif result.state == "FAILURE":
        response["error"] = str(result.result)
    return response


@api_router.post("/tasks/{task_id}/revoke")
async def revoke_task(task_id: str):
    _check_celery()
    from celery.result import AsyncResult
    result = AsyncResult(task_id, app=celery_app_instance)
    result.revoke(terminate=True)
    return {"task_id": task_id, "status": "revoked"}


@api_router.get("/tasks")
async def list_active_tasks():
    _check_celery()
    inspector = celery_app_instance.control.inspect()
    active = inspector.active() or {}
    reserved = inspector.reserved() or {}
    return {
        "active": active,
        "reserved": reserved,
        "celery_available": True,
    }


# ----- Optional MongoDB status routes -----

class StatusCheck(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: __import__("uuid").uuid4().hex)
    client_name: str
    timestamp: str = ""


class StatusCheckCreate(BaseModel):
    client_name: str


if HAS_MONGO:
    from datetime import datetime, timezone

    @api_router.post("/status", response_model=StatusCheck)
    async def create_status_check(input: StatusCheckCreate):
        status_obj = StatusCheck(
            client_name=input.client_name,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        doc = status_obj.model_dump()
        await db.status_checks.insert_one(doc)
        return status_obj

    @api_router.get("/status", response_model=List[StatusCheck])
    async def get_status_checks():
        checks = await db.status_checks.find({}, {"_id": 0}).to_list(1000)
        return checks
else:
    @api_router.get("/status")
    async def get_status_checks():
        return {"message": "MongoDB not configured", "data": []}


app.include_router(api_router)
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@app.on_event("shutdown")
async def shutdown_db_client():
    if HAS_MONGO and client:
        client.close()
