from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from _Adapters import (
    LABELS_4,
    BaseTurnModel,
    Confusion4,
    EasyTurnWP,
    FireRedChatWP,
    NAMOTurnWP,
    SmartTurnWP,
    TENTurnWP,
    get_audio_duration_sec,
    gpu_peak_mem_mb,
    load_dataset,
    normalize_label,
    now_str,
    reset_gpu_peak_mem,
)

# 只对 wait(=dismissal) 做“按文件拆分统计”的两个来源（stem）
WAIT_SPLIT_SRC_STEMS = {
    "SystemIdle_Dismiss_Dismissal",
    "SystemSpeaking_StopandListen_Dismissal",
}

# 按语言定义模型执行顺序（只保留指定 5 个模型）
MODEL_ORDER_BY_LANG: Dict[str, List[str]] = {
    "zh": ["easy_turn", "smart_turn", "ten_turn", "firered", "namo"],
    "en": ["smart_turn", "ten_turn", "firered", "namo"],
}


def existing_files(paths: List[str]) -> List[str]:
    return [p for p in paths if Path(p).exists()]


def save_markdown_report(results: List[Dict[str, Any]], out_path: str):
    if not results:
        print("No results.")
        return
    headers = list(results[0].keys())

    md = []
    md.append("# Turn Detection Benchmark Report\n")
    md.append(f"**Test Date:** {now_str()}\n")
    md.append("| " + " | ".join(headers) + " |")
    md.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in results:
        md.append("| " + " | ".join(str(r.get(h, "")) for h in headers) + " |")
    md.append(
        "\n> 说明：做法A：只在模型支持的 GT 子集上评测（GT 不支持则跳过）。"
        "Latency 统计 warmup 后前 N 次平均值。"
        "GT 优先按最后 assistant 的倒数第二个 tag(Action) 做 action->label4；取不到再 fallback 语义 tag。"
    )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("\n".join(md), encoding="utf-8")
    print(f"\n📄 Report saved to: {out_path}")


def save_json(results: List[Dict[str, Any]], out_path: str):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"🧾 JSON saved to: {out_path}")


def make_run_dir(out_dir: str, lang: str, dataset_tag: str, run_name: str, model_names: List[str]) -> Path:
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    rn = run_name.strip() if run_name else "+".join([m.replace(" ", "").replace("/", "_") for m in model_names])[:80]
    run_dir = out_root / f"{ts}_{lang}_{dataset_tag}_{rn}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def run_benchmark(
    model: BaseTurnModel,
    dataset: List[Dict[str, Any]],
    warmup_iters: int = 20,
    latency_first_n: int = 100,
    save_per_sample_path: Optional[str] = None,
) -> Dict[str, Any]:
    print(f"\n🚀 Model: {model.model_name}")
    print(
        f"  supports_audio={model.supports_audio}, supports_text={model.supports_text}, "
        f"supports_context={model.supports_context}"
    )
    print(f"  supported_labels={sorted(model.supported_labels)}")

    if dataset:
        print(f"-> warmup {warmup_iters} iters ...")
        try:
            model.warmup(dataset[0], warmup_iters)
        except Exception as e:
            print(f"[WARN] warmup failed: {e}")

    reset_gpu_peak_mem()

    conf = Confusion4()
    latencies: List[float] = []
    rtfs: List[float] = []
    skipped = 0
    skipped_unsupported_gt = 0

    per_sample_f = None
    if save_per_sample_path:
        Path(save_per_sample_path).parent.mkdir(parents=True, exist_ok=True)
        per_sample_f = open(save_per_sample_path, "w", encoding="utf-8")

    def write_sample(obj: Dict[str, Any]):
        if per_sample_f:
            per_sample_f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    for idx, s in enumerate(dataset):
        gt_raw = s.get("gt")
        gt_action = s.get("gt_action")
        lang = (s.get("lang") or "").strip() or "zh"
        last_text = s.get("last_text") or ""
        last_wav = s.get("last_wav")
        src = s.get("_src")

        gt = normalize_label(gt_raw)

        if gt == "" or gt not in LABELS_4:
            skipped += 1
            write_sample(
                {
                    "idx": idx,
                    "model": model.model_name,
                    "status": "skipped_invalid_gt",
                    "gt": gt_raw,
                    "gt_norm": gt,
                    "gt_action": gt_action,
                    "lang": lang,
                    "src": src,
                }
            )
            continue

        if gt not in model.supported_labels:
            skipped += 1
            skipped_unsupported_gt += 1
            write_sample(
                {
                    "idx": idx,
                    "model": model.model_name,
                    "status": "skipped_unsupported_gt",
                    "gt": gt_raw,
                    "gt_norm": gt,
                    "gt_action": gt_action,
                    "lang": lang,
                    "src": src,
                }
            )
            continue

        t0 = time.perf_counter()
        try:
            pred_raw = model.predict(s)
            err = None
        except Exception as e:
            pred_raw = None
            err = repr(e)
        t1 = time.perf_counter()

        dt = t1 - t0
        audio_len = get_audio_duration_sec(last_wav or "")
        rtf = dt / max(0.01, audio_len)

        pred = normalize_label(pred_raw)

        if pred == "" or pred not in LABELS_4:
            skipped += 1
            write_sample(
                {
                    "idx": idx,
                    "model": model.model_name,
                    "status": "skipped_invalid_pred" if err is None else "error",
                    "error": err,
                    "gt": gt_raw,
                    "gt_norm": gt,
                    "gt_action": gt_action,
                    "pred": pred_raw,
                    "pred_norm": pred,
                    "lang": lang,
                    "src": src,
                    "latency_s": dt,
                    "rtf": rtf,
                }
            )
            continue

        if pred not in model.supported_labels:
            skipped += 1
            write_sample(
                {
                    "idx": idx,
                    "model": model.model_name,
                    "status": "skipped_unsupported_pred",
                    "gt": gt_raw,
                    "gt_norm": gt,
                    "gt_action": gt_action,
                    "pred": pred_raw,
                    "pred_norm": pred,
                    "lang": lang,
                    "src": src,
                    "latency_s": dt,
                    "rtf": rtf,
                }
            )
            continue

        if len(latencies) < latency_first_n:
            latencies.append(dt)
        rtfs.append(rtf)

        conf.update(pred, gt, lang, src=src, gt_action=gt_action)

        write_sample(
            {
                "idx": idx,
                "model": model.model_name,
                "status": "ok",
                "gt": gt_raw,
                "gt_norm": gt,
                "gt_action": gt_action,
                "pred": pred_raw,
                "pred_norm": pred,
                "correct": bool(pred == gt),
                "lang": lang,
                "src": src,
                "latency_s": dt,
                "audio_len_s": audio_len,
                "rtf": rtf,
                "last_text": last_text,
                "last_wav": last_wav,
            }
        )

    if per_sample_f:
        per_sample_f.close()

    avg_lat_ms = (float(np.mean(latencies)) * 1000.0) if latencies else 0.0
    avg_rtf = float(np.mean(rtfs)) if rtfs else 0.0
    peak_mem = gpu_peak_mem_mb()

    def fmt_pct(x: Optional[float]) -> str:
        return "N/A" if x is None else f"{x*100:.2f}%"

    result: Dict[str, Any] = dict(
        model=model.model_name,
        total=conf.total,
        skipped=skipped,
        skipped_unsupported_gt=skipped_unsupported_gt,
        acc=fmt_pct(conf.acc()),
        acc_complete=fmt_pct(conf.acc_class("complete")),
        acc_incomplete=fmt_pct(conf.acc_class("incomplete")),
        acc_backchannel=fmt_pct(conf.acc_class("backchannel")),
        acc_dismissal=fmt_pct(conf.acc_class("dismissal")),
        acc_takeover=fmt_pct(conf.acc_action("takeover")),
        acc_dismiss=fmt_pct(conf.acc_action("dismiss")),
        acc_maintain=fmt_pct(conf.acc_action("maintain")),
        acc_stopandlisten=fmt_pct(conf.acc_action("stopandlisten")),
        acc_zh=fmt_pct(conf.acc_lang("zh")),
        acc_en=fmt_pct(conf.acc_lang("en")),
        precision_complete=fmt_pct(conf.precision()),
        recall_complete=fmt_pct(conf.recall()),
        f1_complete=fmt_pct(conf.f1()),
        fpr_complete=fmt_pct(conf.fpr()),
        fnr_complete=fmt_pct(conf.fnr()),
        avg_latency_ms=f"{avg_lat_ms:.2f}",
        avg_rtf=f"{avg_rtf:.4f}",
        peak_gpu_mem_mb=f"{peak_mem:.1f}",
    )

    for sname in sorted(WAIT_SPLIT_SRC_STEMS):
        key = f"acc_dismissal__{sname}"
        result[key] = fmt_pct(conf.acc_src_class(sname, "dismissal"))

    print("\n✅ Result:")
    for k in sorted(result.keys()):
        print(f"  {k}: {result[k]}")

    return result


def _pick_lang_path(paths: Dict[str, Dict[str, str]], key: str, lang: str) -> str:
    if key not in paths:
        raise KeyError(f"Missing model path config: {key}")
    if lang not in paths[key]:
        raise KeyError(f"Missing language path config: {key}.{lang}")
    return paths[key][lang]


def _resolve_ten_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def build_models_for_lang(paths: Dict[str, Dict[str, str]], lang: str) -> List[BaseTurnModel]:
    lang = (lang or "").strip().lower()
    if lang not in MODEL_ORDER_BY_LANG:
        raise ValueError(f"Unsupported lang: {lang}")

    ten_device = _resolve_ten_device()

    builders: Dict[str, Callable[[], BaseTurnModel]] = {
        "easy_turn": lambda: EasyTurnWP(paths["easy_turn"]["root"]),
        "smart_turn": lambda: SmartTurnWP(paths["smart_turn"]["onnx"], prefer_gpu=True),
        "ten_turn": lambda: TENTurnWP(paths["ten_turn"]["model"], device=ten_device),
        "firered": lambda: FireRedChatWP(_pick_lang_path(paths, "firered", lang), lang=lang, threshold=0.5, use_gpu=True),
        "namo": lambda: NAMOTurnWP(_pick_lang_path(paths, "namo", lang), filename="model_quant.onnx", prefer_gpu=True),
    }

    models: List[BaseTurnModel] = []
    for model_key in MODEL_ORDER_BY_LANG[lang]:
        if model_key not in builders:
            raise KeyError(f"No builder found for model key: {model_key}")
        models.append(builders[model_key]())

    return models


def run_one_lang(out_dir: str, run_name: str, lang: str, dataset_paths: List[str], paths: Dict[str, Dict[str, str]]):
    dataset: List[Dict[str, Any]] = []
    for p in dataset_paths:
        dataset.extend(load_dataset(p, default_lang=lang))

    print("\n==============================")
    print(f"🧪 Running LANG={lang}  files={len(dataset_paths)}  samples={len(dataset)}")
    print("==============================")

    models_to_run = build_models_for_lang(paths, lang)
    model_names = [m.model_name for m in models_to_run]
    dataset_tag = "+".join([Path(p).stem for p in dataset_paths])[:80]
    run_dir = make_run_dir(out_dir, lang, dataset_tag, run_name, model_names)
    print(f"\n📦 Run output dir: {run_dir}")

    results: List[Dict[str, Any]] = []
    for m in models_to_run:
        per_sample_path = str(run_dir / f"per_sample__{m.model_name.replace('/', '_')}__{lang}.jsonl")
        results.append(run_benchmark(m, dataset, save_per_sample_path=per_sample_path))

    save_markdown_report(results, str(run_dir / "report.md"))
    save_json(results, str(run_dir / "results.json"))

    config = {
        "LANG": lang,
        "datasets": dataset_paths,
        "out_dir": out_dir,
        "run_dir": str(run_dir),
        "run_name": run_name,
        "models": model_names,
        "timestamp": now_str(),
        "paths": paths,
        "note": "GT action uses second last tag of last assistant message; StopandListen always maps to dismissal; report includes per-action accuracy.",
    }
    Path(run_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"🧩 Config saved to: {run_dir / 'config.json'}")


DEFAULT_DATASETS_EN = [
    # "./Benchmark_Datasets/jsonls/EN/syn/syn/SystemIdle_Dismiss_Dismissal.jsonl",
    # "./Benchmark_Datasets/jsonls/EN/syn/syn/SystemIdle_Dismiss_Exclusion.jsonl",
    # "./Benchmark_Datasets/jsonls/EN/syn/syn/SystemIdle_Dismiss_Incomplete.jsonl",
    # "./Benchmark_Datasets/jsonls/EN/syn/syn/SystemIdle_Dismiss_Invalidation.jsonl",
    # "./Benchmark_Datasets/jsonls/EN/syn/syn/SystemIdle_Dismiss_SideTalk.jsonl",
    # "./Benchmark_Datasets/jsonls/EN/syn/syn/SystemIdle_Takeover_Completion.jsonl",
    # "./Benchmark_Datasets/jsonls/EN/syn/syn/SystemIdle_Takeover_Cooperation.jsonl",
    # "./Benchmark_Datasets/jsonls/EN/syn/syn/SystemSpeaking_Maintain_Backchannel.jsonl",
    # "./Benchmark_Datasets/jsonls/EN/syn/syn/SystemSpeaking_Maintain_Distraction.jsonl",
    # "./Benchmark_Datasets/jsonls/EN/syn/syn/SystemSpeaking_Maintain_Invalidation.jsonl",
    # "./Benchmark_Datasets/jsonls/EN/syn/syn/SystemSpeaking_Maintain_SideTalk.jsonl",
    # "./Benchmark_Datasets/jsonls/EN/syn/syn/SystemSpeaking_StopandListen_Collaboration.jsonl",
    # "./Benchmark_Datasets/jsonls/EN/syn/syn/SystemSpeaking_StopandListen_Dismissal.jsonl",
    # "./Benchmark_Datasets/jsonls/EN/syn/syn/SystemSpeaking_StopandListen_Interruption.jsonl",
    # "./Benchmark_Datasets/jsonls/EN/real/test_back2.jsonl",
    # "./Benchmark_Datasets/jsonls/EN/real/test_com2.jsonl",
    # "./Benchmark_Datasets/jsonls/EN/real/test_incom2.jsonl",
    # "./Benchmark_Datasets/jsonls/EN/real/test_inter2.jsonl",
]

DEFAULT_DATASETS_ZH = [
    "./Benchmark_Datasets/jsonls/ZH/syn/syn/SystemIdle_Dismiss_Dismissal.jsonl",
    "./Benchmark_Datasets/jsonls/ZH/syn/syn/SystemIdle_Dismiss_Exclusion.jsonl",
    "./Benchmark_Datasets/jsonls/ZH/syn/syn/SystemIdle_Dismiss_Incomplete.jsonl",
    "./Benchmark_Datasets/jsonls/ZH/syn/syn/SystemIdle_Dismiss_Invalidation.jsonl",
    "./Benchmark_Datasets/jsonls/ZH/syn/syn/SystemIdle_Dismiss_SideTalk.jsonl",
    "./Benchmark_Datasets/jsonls/ZH/syn/syn/SystemIdle_Takeover_Completion.jsonl",
    "./Benchmark_Datasets/jsonls/ZH/syn/syn/SystemIdle_Takeover_Cooperation.jsonl",
    "./Benchmark_Datasets/jsonls/ZH/syn/syn/SystemSpeaking_Maintain_Backchannel.jsonl",
    "./Benchmark_Datasets/jsonls/ZH/syn/syn/SystemSpeaking_Maintain_Distraction.jsonl",
    "./Benchmark_Datasets/jsonls/ZH/syn/syn/SystemSpeaking_Maintain_Invalidation.jsonl",
    "./Benchmark_Datasets/jsonls/ZH/syn/syn/SystemSpeaking_Maintain_SideTalk.jsonl",
    "./Benchmark_Datasets/jsonls/ZH/syn/syn/SystemSpeaking_StopandListen_Collaboration.jsonl",
    "./Benchmark_Datasets/jsonls/ZH/syn/syn/SystemSpeaking_StopandListen_Dismissal.jsonl",
    "./Benchmark_Datasets/jsonls/ZH/syn/syn/SystemSpeaking_StopandListen_Interruption.jsonl",
]


def build_default_paths(base_dir: Path) -> Dict[str, Dict[str, str]]:
    """
    统一模型路径映射：
    - 第一层: 模型 key
    - 第二层: 具体资源或语言键（root/onnx/model/zh/en）
    """
    return {
        "easy_turn": {
            "root": str((base_dir / "Easy_Turn").resolve()),
        },
        "smart_turn": {
            "onnx": str((base_dir / "Smart_Turn_v3/pretrained_models/smart-turn-v3/smart-turn-v3.0.onnx").resolve()),
        },
        "ten_turn": {
            "model": str((base_dir / "TEN_Turn_Detection/pretrained_models/TEN_Turn_Detection").resolve()),
        },
        "firered": {
            "zh": "/FireRedChat/models/FireRedChat-turn-detector",
            "en": "/FireRedChat/models/FireRedChat-turn-detector",
        },
        "namo": {
            "zh": "/NAMO-Turn-Detector-v1/models/Namo-Turn-Detector-v1-Chinese",
            "en": "/NAMO-Turn-Detector-v1/models/Namo-Turn-Detector-v1-English",
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        default="./Turn_Benchamrk/Benchmark",
        help="output root dir",
    )
    parser.add_argument("--run_name", default="", help="optional short name for this run")
    args = parser.parse_args()

    en_paths = existing_files(DEFAULT_DATASETS_EN)
    zh_paths = existing_files(DEFAULT_DATASETS_ZH)

    if not en_paths and not zh_paths:
        raise RuntimeError("No dataset files found. Please check DEFAULT_DATASETS_EN / DEFAULT_DATASETS_ZH paths.")

    base_dir = Path("./Turn_Benchamrk").resolve()
    paths = build_default_paths(base_dir)

    if zh_paths:
        run_one_lang(args.out_dir, args.run_name, "zh", zh_paths, paths)

    if en_paths:
        run_one_lang(args.out_dir, args.run_name, "en", en_paths, paths)


if __name__ == "__main__":
    main()
