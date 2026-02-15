#!/usr/bin/env python
"""
Brute-force interval motif analizi: her eserde tekrarlayan interval n-gram'ları
support ve null-model z-score ile skorlar. Chunk desteği ile paralel çalıştırma.

Kullanım:
  python scripts/brute_force_motif.py                     # tüm eserler, paralel
  python scripts/brute_force_motif.py --chunk 0 --total-chunks 4   # chunk 0/4
  python scripts/brute_force_motif.py --chunk 1 --total-chunks 4   # chunk 1/4
  python scripts/brute_force_motif.py --merge-chunks       # chunk'ları birleştir

Chunk ile: 4 terminalde ayrı ayrı chunk 0,1,2,3 çalıştır → sonra --merge-chunks
"""

from __future__ import annotations

import os
import pandas as pd
from pathlib import Path

DEFAULT_MIN_LENGTH = 4
DEFAULT_MAX_LENGTH = 6
DEFAULT_MIN_SUPPORT = 3
DEFAULT_N_SHUFFLES = 5
DEFAULT_TOP_K = 20


def _process_one_piece(args: tuple) -> tuple | None:
    """Tek eseri işle. ProcessPoolExecutor için picklable."""
    row_dict, root_path, params = args
    root_path = Path(root_path)
    from music_math.data.loader import parse_midi_to_note_events
    from music_math.analysis.pattern_miner import mine_patterns_one_piece

    path = root_path / row_dict["file_path"]
    if not path.exists():
        return None
    try:
        events = parse_midi_to_note_events(path)
        if len(events) < 50:
            return None
        patterns = mine_patterns_one_piece(
            events,
            min_length=params.get("min_length", DEFAULT_MIN_LENGTH),
            max_length=params.get("max_length", DEFAULT_MAX_LENGTH),
            min_support=params.get("min_support", DEFAULT_MIN_SUPPORT),
            n_shuffles=params.get("n_shuffles", DEFAULT_N_SHUFFLES),
            top_k=params.get("top_k", DEFAULT_TOP_K),
        )
        if not patterns:
            return None
        top = patterns[0]
        n_high_z = sum(1 for p in patterns if p["z_score"] >= 2.0)
        summary_row = {
            "filepath": row_dict["file_path"],
            "composer": row_dict.get("composer", ""),
            "era": row_dict.get("era", ""),
            "n_notes": len(events),
            "top_pattern": top["pattern"],
            "top_length": top["length"],
            "top_support": top["support"],
            "top_z_score": top["z_score"],
            "num_patterns_z_ge_2": n_high_z,
        }
        detail_rows = [
            {
                "filepath": row_dict["file_path"],
                "composer": row_dict.get("composer", ""),
                "era": row_dict.get("era", ""),
                "pattern": p["pattern"],
                "length": p["length"],
                "support": p["support"],
                "z_score": p["z_score"],
            }
            for p in patterns[:30]
        ]
        return (summary_row, detail_rows)
    except Exception:
        return None


def run(
    chunk_index: int | None = None,
    total_chunks: int = 1,
    no_parallel: bool = False,
    max_workers: int | None = None,
) -> None:
    from concurrent.futures import ProcessPoolExecutor, as_completed

    from music_math.core.config import CONFIG
    from music_math.core.logging import get_logger
    from music_math.data.loader import parse_midi_to_note_events
    from music_math.analysis.pattern_miner import mine_patterns_one_piece

    logger = get_logger(__name__)
    meta_clean = CONFIG.paths.root / "metadata_clean.csv"
    if not meta_clean.exists():
        raise FileNotFoundError(f"metadata_clean.csv bulunamadı: {meta_clean}")

    stats_dir = CONFIG.paths.root / "results" / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"_chunk{chunk_index}" if chunk_index is not None and total_chunks > 1 else ""
    checkpoint_path = stats_dir / f"brute_force_pattern_summary_checkpoint{suffix}.csv"

    params = {
        "min_length": DEFAULT_MIN_LENGTH,
        "max_length": DEFAULT_MAX_LENGTH,
        "min_support": DEFAULT_MIN_SUPPORT,
        "n_shuffles": DEFAULT_N_SHUFFLES,
        "top_k": DEFAULT_TOP_K,
    }

    df_meta = pd.read_csv(meta_clean)

    done_filepaths = set()
    if checkpoint_path.exists():
        try:
            done_df = pd.read_csv(checkpoint_path)
            done_filepaths = set(done_df["filepath"].astype(str).tolist())
            logger.info("Checkpoint yüklendi: %d eser işlendi (chunk %s)", len(done_filepaths), chunk_index)
        except Exception as e:
            logger.warning("Checkpoint okunamadı: %s", e)

    rows_to_process = []
    for idx, row in df_meta.iterrows():
        if chunk_index is not None and total_chunks > 1 and (idx % total_chunks) != chunk_index:
            continue
        fp = row["file_path"]
        if fp in done_filepaths:
            continue
        path = CONFIG.paths.root / fp
        if not path.exists():
            continue
        rows_to_process.append((idx, row.to_dict(), str(CONFIG.paths.root), params))

    logger.info("İşlenecek eser: %d (chunk %s/%s)", len(rows_to_process), chunk_index, total_chunks)

    summary_rows = []
    detail_rows = []
    if checkpoint_path.exists():
        try:
            prev = pd.read_csv(checkpoint_path)
            summary_rows = prev.to_dict("records")
        except Exception:
            pass

    detail_path = stats_dir / f"brute_force_patterns_detail{suffix}.csv"
    if detail_path.exists() and summary_rows:
        try:
            detail_df = pd.read_csv(detail_path)
            detail_rows = detail_df.to_dict("records")
        except Exception:
            pass

    def _run_serial():
        nonlocal summary_rows, detail_rows
        for idx, row_dict, root_str, prm in rows_to_process:
            path = Path(root_str) / row_dict["file_path"]
            try:
                events = parse_midi_to_note_events(path)
                if len(events) < 50:
                    continue
                patterns = mine_patterns_one_piece(
                    events, **{k: prm[k] for k in ("min_length", "max_length", "min_support", "n_shuffles", "top_k")}
                )
                if not patterns:
                    continue
                top = patterns[0]
                n_high_z = sum(1 for p in patterns if p["z_score"] >= 2.0)
                sr = {
                    "filepath": row_dict["file_path"],
                    "composer": row_dict.get("composer", ""),
                    "era": row_dict.get("era", ""),
                    "n_notes": len(events),
                    "top_pattern": top["pattern"],
                    "top_length": top["length"],
                    "top_support": top["support"],
                    "top_z_score": top["z_score"],
                    "num_patterns_z_ge_2": n_high_z,
                }
                summary_rows.append(sr)
                detail_rows.extend(
                    {
                        "filepath": row_dict["file_path"],
                        "composer": row_dict.get("composer", ""),
                        "era": row_dict.get("era", ""),
                        "pattern": p["pattern"],
                        "length": p["length"],
                        "support": p["support"],
                        "z_score": p["z_score"],
                    }
                    for p in patterns[:30]
                )
                pd.DataFrame(summary_rows).to_csv(checkpoint_path, index=False)
            except Exception as e:
                logger.warning("Hata %s: %s", path, e)

    def _run_parallel():
        nonlocal summary_rows, detail_rows
        max_w = max_workers or max(1, (os.cpu_count() or 4) - 1)
        task_args = [(row_dict, root_str, prm) for _, row_dict, root_str, prm in rows_to_process]
        with ProcessPoolExecutor(max_workers=max_w) as ex:
            futures = {ex.submit(_process_one_piece, ta): ta for ta in task_args}
            done_count = len(summary_rows)
            for fut in as_completed(futures):
                result = fut.result()
                if result is None:
                    continue
                sr, drs = result
                summary_rows.append(sr)
                detail_rows.extend(drs)
                done_count += 1
                pd.DataFrame(summary_rows).to_csv(checkpoint_path, index=False)
                if done_count % 50 == 0:
                    logger.info("Checkpoint: %d eser tamamlandı", done_count)

    if no_parallel:
        _run_serial()
    else:
        _run_parallel()

    df_summary = pd.DataFrame(summary_rows)
    out_summary = stats_dir / f"brute_force_pattern_summary{suffix}.csv"
    df_summary.to_csv(out_summary, index=False)
    logger.info("Özet yazıldı: %s (%d eser)", out_summary, len(df_summary))

    if detail_rows:
        pd.DataFrame(detail_rows).to_csv(detail_path, index=False)

    if "era" in df_summary.columns and not df_summary.empty:
        print("\nEra bazında ortalama top_z_score ve num_patterns_z_ge_2:")
        print(
            df_summary.groupby("era")[["top_z_score", "num_patterns_z_ge_2"]]
            .mean()
            .round(4)
            .to_string()
        )


def merge_chunks(total_chunks: int) -> None:
    """Tüm chunk checkpoint'larını birleştir."""
    from music_math.core.config import CONFIG
    from music_math.core.logging import get_logger

    logger = get_logger(__name__)
    stats_dir = CONFIG.paths.root / "results" / "stats"

    all_summary = []
    all_detail = []
    for i in range(total_chunks):
        cp = stats_dir / f"brute_force_pattern_summary_checkpoint_chunk{i}.csv"
        dp = stats_dir / f"brute_force_patterns_detail_chunk{i}.csv"
        if cp.exists():
            df = pd.read_csv(cp)
            all_summary.append(df)
        if dp.exists():
            df = pd.read_csv(dp)
            all_detail.append(df)

    if not all_summary:
        logger.warning("Birleştirilecek chunk bulunamadı")
        return

    df_summary = pd.concat(all_summary, ignore_index=True).drop_duplicates(subset=["filepath"], keep="last")
    df_summary.to_csv(stats_dir / "brute_force_pattern_summary.csv", index=False)
    logger.info("Birleştirildi: %s (%d eser)", stats_dir / "brute_force_pattern_summary.csv", len(df_summary))

    if all_detail:
        df_detail = pd.concat(all_detail, ignore_index=True)
        df_detail.to_csv(stats_dir / "brute_force_patterns_detail.csv", index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Brute-force motif analizi")
    parser.add_argument("--chunk", type=int, default=None, help="Chunk indeksi (0..total-chunks-1)")
    parser.add_argument("--total-chunks", type=int, default=1, help="Toplam chunk sayısı")
    parser.add_argument("--no-parallel", action="store_true")
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--merge-chunks", action="store_true", help="Chunk'ları birleştir")
    args = parser.parse_args()

    if args.merge_chunks:
        merge_chunks(args.total_chunks)
    else:
        run(
            chunk_index=args.chunk,
            total_chunks=args.total_chunks,
            no_parallel=args.no_parallel,
            max_workers=args.max_workers,
        )
