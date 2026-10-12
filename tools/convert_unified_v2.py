#!/usr/bin/env python
"""Convert the ``search_jev_unified_v2`` release into the SearchJev-V1 per-dataset JEV layout.

The release (``data/search_jev_unified_v2_release_20260928``) stores model rows
(``state / questions / answers / difficulty``) in ``{train,val,test}.jsonl`` and their
metadata (``sample_id``, ``source``, ``task_family`` ...) in line-aligned
``{split}_index.jsonl``.  The training pipeline (``training/data.py``) reads
``<root>/<family>/<dataset>.jsonl`` rows that carry ``sample_id``,
``dataset`` and ``split`` themselves, so the release is converted **once**::

    data/SearchJev-V2/unified_v2/<dataset>.jsonl        one file per release source, all splits
    data/SearchJev-V2/unified_v2/conversion_manifest.json / MANIFEST.md

Row rules (see the release README):

* ``split`` = the file the row came from (``train`` / ``val`` / ``test``), never the index's
  ``source_split`` (that is the upstream split; the test file holds 55k rows whose upstream
  split was train/val).
* ``dataset`` = ``index.source`` with ``/`` -> ``__`` (``retrieval/atec`` -> ``retrieval__atec``):
  no ``/`` (metric keys are ``<dataset>/<qtype>``) and no collision with the 38 SearchJev-V1
  datasets of the same upstream name, so the old wandb keys keep their names.
* ``score`` answers: the release stores the *expected* level (``2.97``); the JEV schema needs
  the integer level, so ``score`` becomes ``argmax(probabilities)`` (== the index ``gold``,
  verified for every question) and the calibrated ``probabilities`` stay as the soft target.
  The original value is kept as ``expected_score``.
* ``meta`` keeps the short index fields (source, task_family, capability, training_lane,
  label_tier, supervision, weak, source_split, difficulty, language_heuristic, input_origin).

CPU only, streaming, idempotent (writes ``*.tmp`` then renames; ``--force`` overwrites).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RELEASE = PROJECT_ROOT / "data" / "search_jev_unified_v2_release_20260928"
DEFAULT_OUT = PROJECT_ROOT / "data" / "SearchJev-V2" / "unified_v2"
CONVERTER_VERSION = 1
SPLITS = ("train", "val", "test")
META_KEYS = ("source", "task_family", "capability", "training_lane", "label_tier", "supervision", "weak",
             "source_split", "difficulty", "language_heuristic", "input_origin")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def dataset_name(source: str) -> str:
    return source.strip().replace("/", "__")


def fix_score_answer(answer: dict, criteria: list, gold_level) -> tuple[dict, str]:
    """Return the answer with an integer ``score`` level and the fix kind applied."""
    out = dict(answer)
    raw = answer.get("score")
    probs = answer.get("probabilities")
    if isinstance(probs, dict) and probs:
        try:
            level = min(int(k) for k in probs if float(probs[k]) == max(float(v) for v in probs.values()))
        except (TypeError, ValueError):
            level = None
    else:
        level = None
    if level is None:
        if raw is None:
            return out, "missing"
        level = int(round(float(raw)))
        kind = "rounded"
    else:
        kind = "argmax"
    if not (0 <= level < len(criteria)):
        return out, "out_of_range"
    if raw is not None and (not isinstance(raw, (int, float)) or float(raw) != float(level)):
        out["expected_score"] = raw
    out["score"] = level
    if gold_level is not None and int(gold_level) != level:
        kind += "|gold_mismatch"
    return out, kind


def convert(release: Path, out_dir: Path, force: bool = False, verify: bool = True) -> dict:
    manifest = json.loads((release / "manifest.json").read_text(encoding="utf-8"))
    if "search_jev_unified_v2" not in str(manifest.get("name", "")) or "rows" not in manifest:
        raise SystemExit(f"{release} does not look like the unified v2 release (manifest.json)")
    out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / "conversion_manifest.json"
    if final.exists() and not force:
        raise SystemExit(f"{out_dir} already converted ({final}); use --force to redo")
    t0 = time.time()
    input_sha = {}
    for split in SPLITS:
        for name in (f"{split}.jsonl", f"{split}_index.jsonl"):
            want = manifest["sha256"][name]
            if verify:
                got = sha256_of(release / name)
                if got != want:
                    raise SystemExit(f"sha256 mismatch for {name}: {got} != manifest {want}")
            input_sha[name] = want
    print(f"inputs verified in {time.time() - t0:.0f}s" if verify else "inputs not verified (--no-verify)", flush=True)

    handles: dict[str, object] = {}
    counts: dict[str, Counter] = defaultdict(Counter)          # dataset -> split -> rows
    qcounts: dict[str, Counter] = defaultdict(Counter)         # dataset -> split -> questions
    qtypes: dict[str, Counter] = defaultdict(Counter)          # dataset -> qtype -> questions
    task_of: dict[str, str] = {}
    fixes: Counter = Counter()
    gold_mismatch: Counter = Counter()
    per_split_rows: Counter = Counter()
    ids_seen: set[str] = set()
    dup_ids = 0

    def handle(ds: str):
        fh = handles.get(ds)
        if fh is None:
            fh = (out_dir / f"{ds}.jsonl.tmp").open("w", encoding="utf-8")
            handles[ds] = fh
        return fh

    for split in SPLITS:
        n_expected = int(manifest["rows"][split])
        n = 0
        with (release / f"{split}.jsonl").open(encoding="utf-8") as fd, \
                (release / f"{split}_index.jsonl").open(encoding="utf-8") as fi:
            for ld, li in zip(fd, fi, strict=True):
                r = json.loads(ld)
                ix = json.loads(li)
                n += 1
                src = str(ix["source"])
                ds = dataset_name(src)
                sid = str(ix["sample_id"])
                if sid in ids_seen:
                    dup_ids += 1
                ids_seen.add(sid)
                gold = ix.get("gold")
                if isinstance(gold, str):
                    try:
                        gold = json.loads(gold)
                    except ValueError:
                        gold = {}
                gold = gold if isinstance(gold, dict) else {}
                questions = r["questions"]
                answers = dict(r.get("answers") or {})
                for qid, q in questions.items():
                    qtypes[ds][q.get("type")] += 1
                    if q.get("type") == "score" and qid in answers:
                        answers[qid], kind = fix_score_answer(answers[qid], q.get("criteria") or [], gold.get(qid))
                        fixes[kind.split("|")[0]] += 1
                        if "gold_mismatch" in kind:
                            gold_mismatch[ds] += 1
                row = {
                    "sample_id": sid,
                    "dataset": ds,
                    "task": f"{ix.get('task_family', '')}/{ix.get('capability', '')}",
                    "split": split,
                    "state": r["state"],
                    "questions": questions,
                    "answers": answers,
                    "meta": {k: ix.get(k) for k in META_KEYS if ix.get(k) is not None},
                }
                if r.get("difficulty") is not None:
                    row["meta"]["difficulty"] = r["difficulty"]
                handle(ds).write(json.dumps(row, ensure_ascii=False) + "\n")
                counts[ds][split] += 1
                qcounts[ds][split] += len(questions)
                task_of.setdefault(ds, str(ix.get("task_family", "")))
                per_split_rows[split] += 1
        if n != n_expected:
            raise SystemExit(f"{split}: read {n} rows, manifest says {n_expected}")
        print(f"{split}: {n:,} rows -> {len(handles)} dataset files so far ({time.time() - t0:.0f}s)", flush=True)

    for fh in handles.values():
        fh.close()
    for ds in handles:
        (out_dir / f"{ds}.jsonl.tmp").replace(out_dir / f"{ds}.jsonl")
    # remove stale files from an earlier conversion with different names
    for p in out_dir.glob("*.jsonl"):
        if p.stem not in handles:
            p.unlink()

    datasets = {}
    for ds in sorted(handles):
        datasets[ds] = {
            "source": ds.replace("__", "/", 1), "task_family": task_of[ds],
            "rows": {s: counts[ds].get(s, 0) for s in SPLITS},
            "questions": {s: qcounts[ds].get(s, 0) for s in SPLITS},
            "qtypes": dict(qtypes[ds]), "score_gold_mismatch": gold_mismatch.get(ds, 0),
            "sha256": sha256_of(out_dir / f"{ds}.jsonl"),
        }
    report = {
        "converter_version": CONVERTER_VERSION, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "release_dir": str(release), "release_manifest": {k: manifest[k] for k in ("name", "release_date", "updated", "rows")},
        "input_sha256": input_sha, "rows_per_split": dict(per_split_rows), "n_datasets": len(datasets),
        "duplicate_sample_ids": dup_ids, "score_fixes": dict(fixes), "score_gold_mismatch_total": sum(gold_mismatch.values()),
        "datasets": datasets, "seconds": time.time() - t0,
    }
    (out_dir / "conversion_manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    lines = [f"# unified_v2 family (converted from `{release.name}`)", "",
             f"Converted {time.strftime('%Y-%m-%d %H:%M')} by `tools/convert_unified_v2.py` v{CONVERTER_VERSION}; "
             f"release `{manifest.get('name')}` {manifest.get('release_date')} (updated {manifest.get('updated')}).",
             "", "One file per release source; rows carry `sample_id`, `dataset` (= source with `/` -> `__`), `task`, `split` "
             "(= release file), `state`, `questions`, `answers` (score = integer level = argmax of the calibrated "
             "`probabilities`; original expected value kept as `expected_score`), `meta` (index fields).", "",
             f"Rows: train {per_split_rows['train']:,} / val {per_split_rows['val']:,} / test {per_split_rows['test']:,}; "
             f"score fixes {dict(fixes)}; gold mismatches {sum(gold_mismatch.values())}.", "",
             "| dataset | task_family | train | val | test | questions (t/v/te) | qtypes |", "|---|---|---:|---:|---:|---|---|"]
    for ds, d in datasets.items():
        lines.append(f"| {ds} | {d['task_family']} | {d['rows']['train']:,} | {d['rows']['val']:,} | {d['rows']['test']:,} | "
                     f"{d['questions']['train']:,}/{d['questions']['val']:,}/{d['questions']['test']:,} | "
                     f"{', '.join(f'{k}={v}' for k, v in sorted(d['qtypes'].items()))} |")
    (out_dir / "MANIFEST.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--release", default=str(DEFAULT_RELEASE))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--force", action="store_true", help="overwrite an existing conversion")
    p.add_argument("--no-verify", action="store_true", help="skip the sha256 check of the release files")
    a = p.parse_args(argv)
    rep = convert(Path(a.release), Path(a.out), force=a.force, verify=not a.no_verify)
    print(json.dumps({k: rep[k] for k in ("rows_per_split", "n_datasets", "duplicate_sample_ids", "score_fixes",
                                          "score_gold_mismatch_total", "seconds")}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
