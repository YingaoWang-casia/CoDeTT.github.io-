
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ========= Config =========

# ✅ 本地模型路径
LOCAL_MODEL_DIR = "./Turn_Benchamrk/KE-SemanticVAD"

# ✅ HF 仓库名作为 fallback（本地不存在时才会用）
HF_MODEL_ID = "KE-Team/KE-SemanticVAD"

LABELS_4 = ["complete", "incomplete", "backchannel", "dismissal"]  # dismissal == wait
TAG_RE = re.compile(r"<\|([^|]+)\|>")


def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def existing_files(paths: List[str]) -> List[str]:
    return [p for p in paths if Path(p).exists()]


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    stem = Path(path).stem
    for s in data:
        s["_src"] = stem
    return data


def save_json(obj: Any, out_path: str):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"🧾 JSON saved to: {out_path}")


def save_markdown_report(rows: List[Dict[str, Any]], out_path: str):
    if not rows:
        print("No results.")
        return
    headers = list(rows[0].keys())
    md = []
    md.append("# Turn Detection Benchmark Report\n")
    md.append(f"**Test Date:** {now_str()}\n")
    md.append("| " + " | ".join(headers) + " |")
    md.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        md.append("| " + " | ".join(str(r.get(h, "")) for h in headers) + " |")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("\n".join(md), encoding="utf-8")
    print(f"\n📄 Report saved to: {out_path}")


def fmt_pct(x: Optional[float]) -> str:
    return "N/A" if x is None else f"{x*100:.2f}%"


# ========= KE prompts（按你提供的原版） =========

AGNET_SPKING_SYS = (
    "# Role\n你是人机实时交互的**用户行为分析**模块，你将收到包含部分历史信息的 Human 和 Agent 最新实时对话记录 (Dialog)\n\n"
    "# 任务\n当前【Agent正在发言】，在此过程中，你需要基于对话分析 Human 的意图属于 <打断> 还是 <附和>\n\n"
    "# 输出\n不要有多余的分析，仅严格输出以下二者之一: <打断> 或 <附和>\n\n"
    "# 判断标准\n## <打断> 的情况\nHuman 行为: 试图抢夺话题主导权\n特征包括:\n"
    "- 提供新概念/词汇/判断（如命名、定性、对比）\n- 提出问题或异议\n- 引入与当前话题无关的新话题\n\n"
    "## <附和> 的情况\nHuman 行为: 赞同 Agent, 期望 Agent 继续说\n特征包括:\n"
    "- 使用零内容反馈（嗯/啊/对）\n- 机械重复 Agent 中的原词/同义词\n- 表达简单的确认或同意（如“是的”、“没错”）\n"
)

HUMAN_SPKING_SYS = (
    "# Role\n你是人机实时交互的**用户行为分析**模块，你将收到包含部分历史信息的 Human 和 Agent 最新实时对话记录 (Dialog)\n\n"
    "# 任务\n当前【Human正在发言】，你需要基于对话判断 Human 是否已经完成发言\n\n"
    "# 输出\n严格输出以下二者之一: <完成> 或 <未完>\n\n"
    "# 判断标准\n## <完成> 的情况\nHuman 发言语义完整，说话很可能已经结束\n"
    "- 发言包含完整命题（如明确提问/请求/结论）\n- 出现结束性标记词（\"好了\"/\"你觉得呢？\"）\n\n"
    "## <未完> 的情况\nHuman 发言语义不完整，仍然可能继续说话\n"
    "- 语句末尾含连接词（\"而且\"/\"不过\"/\"然后\"）\n- 用户发言中夹杂思考词（\"呃...\"/\"嗯...\"）\n"
)


# ========= 标签解析/映射 =========

def extract_last_semantic_tag(text: str) -> str:
    if not text:
        return ""
    tags = TAG_RE.findall(text)
    return tags[-1] if tags else ""


def semantic_tag_to_label4(tag: str) -> str:
    t = (tag or "").strip().lower()
    if t == "completion":
        return "complete"
    if t == "incomplete":
        return "incomplete"
    if t == "backchannel":
        return "backchannel"
    if t == "dismissal":
        return "dismissal"
    if t == "wait":
        return "dismissal"
    return ""


def get_gt_label4(sample: Dict[str, Any]) -> str:
    """
    GT: 最后一条 assistant content 形如 "<|Takeover|><|Completion|>"
    只取最后一个标签 => Completion/Incomplete/Backchannel/Dismissal
    """
    msgs = sample.get("messages")
    if not isinstance(msgs, list) or not msgs:
        return ""
    last = msgs[-1]
    if not isinstance(last, dict):
        return ""
    if (last.get("role") or "").strip() != "assistant":
        return ""
    content = str(last.get("content") or "")
    tag = extract_last_semantic_tag(content)
    return semantic_tag_to_label4(tag)


# ========= 历史轮数截断 =========

def truncate_by_rounds(clean_msgs: List[Dict[str, str]], history_rounds: int) -> List[Dict[str, str]]:
    """
    语义（按你要求）：
      - history_rounds = 0: 只保留当前句
      - history_rounds = 1: 当前句 + 往上 2 句
      - history_rounds = 2: 当前句 + 往上 4 句
      - ...
      - history_rounds < 0: 不截断（全量历史），用于对比实验
    """
    if not clean_msgs:
        return clean_msgs

    # 负数：保留全量历史（可选后门）
    if history_rounds is not None and history_rounds < 0:
        return clean_msgs

    # 拆出 system（如果有）
    sys_msg = None
    start_idx = 0
    if clean_msgs and clean_msgs[0].get("role") == "system":
        sys_msg = clean_msgs[0]
        start_idx = 1

    non_sys = clean_msgs[start_idx:]
    if not non_sys:
        return [sys_msg] if sys_msg is not None else []

    # ✅ 核心：只保留“当前句 + 2*history_rounds 条历史”
    h = 0 if history_rounds is None else max(0, int(history_rounds))
    keep_n = 1 + 2 * h
    tail = non_sys[-keep_n:] if len(non_sys) > keep_n else non_sys

    return ([sys_msg] + tail) if sys_msg is not None else tail


def strip_gt_and_wav(sample: Dict[str, Any], history_rounds: int = 0) -> List[Dict[str, str]]:
    """
    推理输入：去掉最后一条 GT tag；只保留 role/content（audio、wav_path等字段不会进入模型）
    history_rounds: 保留多少轮历史（按 truncate_by_rounds 的语义）
    """
    msgs = sample.get("messages")
    if not isinstance(msgs, list) or not msgs:
        last_text = str(sample.get("last_text") or sample.get("query") or "")
        base = [
            {"role": "system", "content": "You are a turn-taking decision model."},
            {"role": "user", "content": last_text},
        ]
        return truncate_by_rounds(base, history_rounds)

    clean: List[Dict[str, str]] = []
    for m in msgs:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role") or "").strip()
        content = str(m.get("content") or "")
        if role and content:
            # ✅ 只取 role/content，不会把 audio 路径喂进去
            clean.append({"role": role, "content": content})

    # 去掉最后一条 GT assistant tag
    if clean and clean[-1]["role"] == "assistant" and "<|" in clean[-1]["content"]:
        clean = clean[:-1]

    # ✅ 按轮数截断历史
    clean = truncate_by_rounds(clean, history_rounds)
    return clean


def build_dialog_for_ke(messages_wo_gt: List[Dict[str, str]], mode: str) -> List[Dict[str, str]]:
    """
    mode:
      - "user"  => HUMAN_SPKING_SYS  => 输出 <完成>/<未完>
      - "agent" => AGNET_SPKING_SYS  => 输出 <打断>/<附和>
    """
    system = HUMAN_SPKING_SYS if mode == "user" else AGNET_SPKING_SYS

    lines = ["# Dialog"]
    for m in messages_wo_gt:
        r = m["role"]
        c = m["content"]
        if r == "user":
            lines.append(f"Human:{c}")
        elif r == "assistant":
            lines.append(f"Agent:{c}")
    dialog_text = "\n".join(lines) + "\n"

    return [{"role": "system", "content": system}, {"role": "user", "content": dialog_text}]


def ke_output_to_label4(raw: str, mode: str) -> str:
    """
    mode=user:  <完成>/<未完> -> complete/incomplete
    mode=agent: <附和>/<打断> -> backchannel/dismissal(wait)
    """
    s = (raw or "").strip()
    if mode == "agent":
        if "打断" in s:
            return "dismissal"  # == wait
        if "附和" in s:
            return "backchannel"
        # 兜底：有些模型可能输出英文
        if "<interrupt>" in s or "interrupt" in s.lower():
            return "dismissal"
        if "<backchannel>" in s or "backchannel" in s.lower():
            return "backchannel"
    else:
        if "完成" in s:
            return "complete"
        if "未完" in s:
            return "incomplete"
        if "<complete>" in s.lower():
            return "complete"
        if "<incomplete>" in s.lower():
            return "incomplete"
    return ""


# ========= KE 模型封装 =========

def resolve_model_path(local_dir: str, hf_id: str) -> str:
    """
    优先使用本地目录；不存在就 fallback 到 HF id
    """
    if local_dir and Path(local_dir).exists():
        return local_dir
    return hf_id


class KESemanticVAD:
    def __init__(self, device: Optional[str] = None, max_new_tokens: int = 4096):
        model_ref = resolve_model_path(LOCAL_MODEL_DIR, HF_MODEL_ID)
        self.model_id = model_ref

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.max_new_tokens = max_new_tokens

        # ✅ 强制 slow tokenizer + trust_remote_code，规避 fast tokenizer 解析报错
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_ref,
            use_fast=False,
            trust_remote_code=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_ref,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def predict_label4(self, sample: Dict[str, Any], mode: str, history_rounds: int = 0) -> Tuple[str, str]:
        msgs_wo_gt = strip_gt_and_wav(sample, history_rounds=history_rounds)
        ke_messages = build_dialog_for_ke(msgs_wo_gt, mode=mode)

        text = self.tokenizer.apply_chat_template(ke_messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        gen_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )
        gen_ids = [out[len(inp):] for inp, out in zip(model_inputs.input_ids, gen_ids)]
        out_text = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]

        pred = ke_output_to_label4(out_text, mode=mode)
        return pred, out_text


# ========= 统计 =========

class Confusion4:
    def __init__(self):
        self.total = 0
        self.correct = 0
        self.by_class_total = {c: 0 for c in LABELS_4}
        self.by_class_correct = {c: 0 for c in LABELS_4}

    def update(self, pred: str, gt: str):
        self.total += 1
        ok = (pred == gt)
        self.correct += int(ok)
        self.by_class_total[gt] += 1
        self.by_class_correct[gt] += int(ok)

    def acc(self) -> Optional[float]:
        return None if self.total == 0 else self.correct / self.total

    def acc_class(self, c: str) -> Optional[float]:
        t = self.by_class_total.get(c, 0)
        return None if t == 0 else self.by_class_correct.get(c, 0) / t


def make_run_dir(out_dir: str, dataset_tag: str, run_name: str) -> Path:
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    rn = run_name.strip() if run_name else "KE-SemanticVAD"
    run_dir = out_root / f"{ts}_{dataset_tag}_{rn}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def run_benchmark(
    model: KESemanticVAD,
    dataset: List[Dict[str, Any]],
    force_mode: str,  # "user" or "agent"
    warmup_iters: int = 10,
    latency_first_n: int = 100,
    save_per_sample_path: Optional[str] = None,
    history_rounds: int = 0,
) -> Dict[str, Any]:
    print(
        f"\n🚀 Model: {model.model_id} device={model.device}  force_mode={force_mode}  history_rounds={history_rounds}"
    )

    if dataset:
        print(f"-> warmup {warmup_iters} iters ...")
        for _ in range(max(1, warmup_iters)):
            try:
                _ = model.predict_label4(dataset[0], mode=force_mode, history_rounds=history_rounds)
            except Exception:
                pass

    conf = Confusion4()
    latencies: List[float] = []
    skipped = 0
    errors = 0

    per_sample_f = None
    if save_per_sample_path:
        Path(save_per_sample_path).parent.mkdir(parents=True, exist_ok=True)
        per_sample_f = open(save_per_sample_path, "w", encoding="utf-8")

    def write_sample(obj: Dict[str, Any]):
        if per_sample_f:
            per_sample_f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    for idx, s in enumerate(dataset):
        gt = get_gt_label4(s)
        src = s.get("_src")

        if gt not in LABELS_4:
            skipped += 1
            write_sample({"idx": idx, "status": "skipped_invalid_gt", "gt": gt, "src": src})
            continue

        t0 = time.perf_counter()
        try:
            pred, raw = model.predict_label4(s, mode=force_mode, history_rounds=history_rounds)
            err = None
        except Exception as e:
            pred, raw = "", ""
            err = repr(e)
        t1 = time.perf_counter()

        dt = t1 - t0
        if len(latencies) < latency_first_n:
            latencies.append(dt)

        if err is not None:
            errors += 1
            write_sample(
                {"idx": idx, "status": "error", "error": err, "gt": gt, "pred": pred, "raw": raw, "src": src}
            )
            continue

        if pred not in LABELS_4:
            skipped += 1
            write_sample(
                {
                    "idx": idx,
                    "status": "skipped_invalid_pred",
                    "gt": gt,
                    "pred": pred,
                    "raw": raw,
                    "src": src,
                    "latency_s": dt,
                }
            )
            continue

        conf.update(pred, gt)
        write_sample(
            {
                "idx": idx,
                "status": "ok",
                "gt": gt,
                "pred": pred,
                "correct": bool(pred == gt),
                "raw": raw,
                "src": src,
                "latency_s": dt,
            }
        )

    if per_sample_f:
        per_sample_f.close()

    avg_lat_ms = (float(np.mean(latencies)) * 1000.0) if latencies else 0.0

    result = dict(
        model=str(model.model_id),
        mode=force_mode,
        history_rounds=history_rounds,
        total=conf.total,
        skipped=skipped,
        errors=errors,
        acc=fmt_pct(conf.acc()),
        acc_complete=fmt_pct(conf.acc_class("complete")),
        acc_incomplete=fmt_pct(conf.acc_class("incomplete")),
        acc_backchannel=fmt_pct(conf.acc_class("backchannel")),
        acc_dismissal=fmt_pct(conf.acc_class("dismissal")),  # dismissal == wait
        avg_latency_ms=f"{avg_lat_ms:.2f}",
    )

    print("\n✅ Result:")
    for k in sorted(result.keys()):
        print(f"  {k}: {result[k]}")
    return result


# ========= 默认输入（多文件） =========

DEFAULT_DATASETS_USER = [
    "/Benchmark_Datasets/jsonls/EN/real/SystemIdle_Dismiss_Incomplete.jsonl",
    "/Benchmark_Datasets/jsonls/EN/real/SystemIdle_Takeover_Completion.jsonl",
    "/Benchmark_Datasets/jsonls/EN//syn/SystemIdle_Dismiss_Incomplete.jsonl",
    "/Benchmark_Datasets/jsonls/EN/syn/SystemIdle_Takeover_Completion.jsonl",
]

DEFAULT_DATASETS_AGENT = [
    "/Benchmark_Datasets/jsonls/EN/real/SystemSpeaking_Maintain_Backchannel.jsonl",
    "/Benchmark_Datasets/jsonls/EN/syn/SystemIdle_Dismiss_Dismissal.jsonl",
    "/Benchmark_Datasets/jsonls/EN/syn/SystemSpeaking_Maintain_Backchannel.jsonl",
    "/Benchmark_Datasets/jsonls/EN/syn/SystemSpeaking_StopandListen_Dismissal.jsonl",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="./Turn_Benchamrk/Benchmark_ke")
    parser.add_argument("--run_name", default="")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max_new_tokens", type=int, default=16)

    # ✅ 你要的语义：0=只保留当前句；1=当前句+往上2句；2=当前句+往上4句...
    # ✅ 额外：<0 表示不截断（全量历史）用于对比实验
    parser.add_argument(
        "--history_rounds",
        type=int,
        default=0,
        help="0=only current utterance; 1=+2 prev; 2=+4 prev ...; <0 keep all history.",
    )

    # 仍然支持 CLI 覆盖默认多文件
    parser.add_argument("--datasets_user", nargs="*", default=None, help="override user datasets")
    parser.add_argument("--datasets_agent", nargs="*", default=None, help="override agent datasets")

    args = parser.parse_args()

    ds_user = DEFAULT_DATASETS_USER if args.datasets_user is None else args.datasets_user
    ds_agent = DEFAULT_DATASETS_AGENT if args.datasets_agent is None else args.datasets_agent

    paths_user = existing_files(ds_user)
    paths_agent = existing_files(ds_agent)

    if not paths_user and not paths_agent:
        raise RuntimeError(
            "No dataset files found.\n"
            "Please fill DEFAULT_DATASETS_USER / DEFAULT_DATASETS_AGENT in benchmark.py\n"
            "or pass --datasets_user / --datasets_agent in CLI."
        )

    model = KESemanticVAD(device=args.device, max_new_tokens=args.max_new_tokens)

    results: List[Dict[str, Any]] = []

    dataset_tag_parts = []
    if paths_user:
        dataset_tag_parts.append("USER:" + "+".join([Path(p).stem for p in paths_user])[:80])
    if paths_agent:
        dataset_tag_parts.append("AGENT:" + "+".join([Path(p).stem for p in paths_agent])[:80])
    dataset_tag = "__".join(dataset_tag_parts) if dataset_tag_parts else "run"

    run_dir = make_run_dir(args.out_dir, dataset_tag, args.run_name)
    print(f"\n📦 Run output dir: {run_dir}")

    if paths_user:
        dataset_user: List[Dict[str, Any]] = []
        for p in paths_user:
            dataset_user.extend(read_jsonl(p))
        print(f"\n🧪 USER mode files={len(paths_user)} samples={len(dataset_user)}")
        per_sample_path = str(run_dir / "per_sample__KE-SemanticVAD__user.jsonl")
        results.append(
            run_benchmark(
                model,
                dataset_user,
                force_mode="user",
                save_per_sample_path=per_sample_path,
                history_rounds=args.history_rounds,
            )
        )

    if paths_agent:
        dataset_agent: List[Dict[str, Any]] = []
        for p in paths_agent:
            dataset_agent.extend(read_jsonl(p))
        print(f"\n🧪 AGENT mode files={len(paths_agent)} samples={len(dataset_agent)}")
        per_sample_path = str(run_dir / "per_sample__KE-SemanticVAD__agent.jsonl")
        results.append(
            run_benchmark(
                model,
                dataset_agent,
                force_mode="agent",
                save_per_sample_path=per_sample_path,
                history_rounds=args.history_rounds,
            )
        )

    save_markdown_report(results, str(run_dir / "report.md"))
    save_json(results, str(run_dir / "results.json"))

    config = {
        "datasets_user": paths_user,
        "datasets_agent": paths_agent,
        "out_dir": args.out_dir,
        "run_dir": str(run_dir),
        "run_name": args.run_name,
        "model_local_dir": LOCAL_MODEL_DIR,
        "model_hf_id": HF_MODEL_ID,
        "model_used": str(model.model_id),
        "device": model.device,
        "max_new_tokens": args.max_new_tokens,
        "history_rounds": args.history_rounds,
        "timestamp": now_str(),
        "note": "dismissal == wait",
    }
    Path(run_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"🧩 Config saved to: {run_dir / 'config.json'}")


if __name__ == "__main__":
    main()
