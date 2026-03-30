from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ===== MiniCPM 依赖 =====
import torch
from transformers import AutoModel

# ============================================================
# 配置（本地 MiniCPM-o-4_5）
# ============================================================
LOCAL_MODEL_DIR = "./Turn_Benchamrk/MiniCPM-o-4_5/dir"

# 评测并发：你原来是 16 线程；但本地同一张 GPU 并发推理容易 OOM
# 这里保留 ThreadPool 的 16，用一个全局锁把真正的 model.chat 串行化（稳定优先）
NUM_WORKERS = 16

# ✅ 你要的模式：user 输入音频，assistant 输入文本（history 中 user 用音频，assistant 用文本）
USER_AUDIO_ASSISTANT_TEXT_HISTORY = True

# ✅ 纯音频模式下，如果 history 某条没有 audio，是否直接跳过
SKIP_HISTORY_WITHOUT_AUDIO = True
MAX_NEW_TOKENS = 4096
DO_SAMPLE = False
TEMPERATURE = 0.0

# 历史轮数控制：最近 N 轮（user+assistant 为 1 轮）
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "5"))

# ============================================================
# 14 标签（闭集）- 无修改
# ============================================================
LABEL2TRIPLE: Dict[str, Tuple[str, str, str]] = {
    # SystemSpeaking = 7
    "SystemSpeaking_Maintain_Backchannel": ("SystemSpeaking", "Maintain", "Backchannel"),
    "SystemSpeaking_Maintain_Invalidation": ("SystemSpeaking", "Maintain", "Invalidation"),
    "SystemSpeaking_Maintain_SideTalk": ("SystemSpeaking", "Maintain", "SideTalk"),
    "SystemSpeaking_Maintain_Distraction": ("SystemSpeaking", "Maintain", "Distraction"),
    "SystemSpeaking_StopandListen_Interruption": ("SystemSpeaking", "StopandListen", "Interruption"),
    "SystemSpeaking_StopandListen_Dismissal": ("SystemSpeaking", "StopandListen", "Dismissal"),
    "SystemSpeaking_StopandListen_Collaboration": ("SystemSpeaking", "StopandListen", "Collaboration"),

    # SystemIdle = 7
    "SystemIdle_Takeover_Completion": ("SystemIdle", "Takeover", "Completion"),
    "SystemIdle_Takeover_Cooperation": ("SystemIdle", "Takeover", "Cooperation"),
    "SystemIdle_Dismiss_Incomplete": ("SystemIdle", "Dismiss", "Incomplete"),
    "SystemIdle_Dismiss_Invalidation": ("SystemIdle", "Dismiss", "Invalidation"),
    "SystemIdle_Dismiss_Dismissal": ("SystemIdle", "Dismiss", "Dismissal"),
    "SystemIdle_Dismiss_Exclusion": ("SystemIdle", "Dismiss", "Exclusion"),
    "SystemIdle_Dismiss_SideTalk": ("SystemIdle", "Dismiss", "SideTalk"),
}
LABELS = list(LABEL2TRIPLE.keys())
TRIPLE2LABEL = {v: k for k, v in LABEL2TRIPLE.items()}

LABEL_META: Dict[str, Dict[str, str]] = {
    "SystemSpeaking_Maintain_Backchannel": {
        "zh": "系统在说话，用户短促附和/跟随（嗯/对/好/明白/OK/继续），信息量低，不提新需求，不要求停。"
    },
    "SystemSpeaking_Maintain_Invalidation": {
        "zh": "系统在说话，用户输入为无效声学信号（咳嗽/清嗓/喷嚏/不可辨识碎音/摩擦声/强背景噪声），不构成对话内容。"
    },
    "SystemSpeaking_Maintain_SideTalk": {
        "zh": "系统在说话，用户在跟第三方交流（对象不是系统），如“你先…/给我拿…/别闹…/叫某人名字”。"
    },
    "SystemSpeaking_Maintain_Distraction": {
        "zh": "系统在说话，环境第三方语音/电视广播等，与当前话题无关（远场、多人与混响），应忽略。"
    },
    "SystemSpeaking_StopandListen_Interruption": {
        "zh": "系统在说话，用户明确插话想让系统立刻停下并听（等一下/不对/我想问/你先听我说/我要改一下），信息量显著更高。"
    },
    "SystemSpeaking_StopandListen_Dismissal": {
        "zh": "系统在说话，用户明确要求系统停止当前输出或结束交互（别说了/停/不用了/结束/算了）。"
    },
    "SystemSpeaking_StopandListen_Collaboration": {
        "zh": "系统在说话，第三方（非主用户）介入且与话题相关、能推进任务，系统应停下听并必要时确认。"
    },

    "SystemIdle_Takeover_Completion": {
        "zh": "系统空闲，用户话轮完成（EOU），句子完整、意图清晰，可直接回应。"
    },
    "SystemIdle_Takeover_Cooperation": {
        "zh": "系统空闲，第三方提供对任务有价值的协作信息（时间/地点/偏好/纠错），系统可接管回应并整合。"
    },
    "SystemIdle_Dismiss_Incomplete": {
        "zh": "系统空闲，用户未说完/思考中/半句停住/自我修正（那个…/我想…/等下…），系统不抢答继续等。"
    },
    "SystemIdle_Dismiss_Invalidation": {
        "zh": "系统空闲，用户输入为无效声学信号（咳嗽/噪声/不可辨识），系统应忽略并继续等待。"
    },
    "SystemIdle_Dismiss_Dismissal": {
        "zh": "系统空闲，用户撤销或终止需求（算了/不用了/结束/退出），系统不进入长回复。"
    },
    "SystemIdle_Dismiss_Exclusion": {
        "zh": "系统空闲，用户语用上排除系统参与（不是跟你说的/你别管/没叫你/我问他呢），系统不介入。"
    },
    "SystemIdle_Dismiss_SideTalk": {
        "zh": "系统空闲，用户对第三方说话（未显式排除系统），不要求系统响应。"
    },
}

# ============================================================
# 工具函数
# ============================================================
def normalize_label(x: Any) -> str:
    return str(x).strip() if x is not None else ""

def normalize_token_tag(x: str) -> str:
    if not x:
        return ""
    s = str(x).replace("<|", "").replace("|>", "").strip()
    return s[:1].upper() + s[1:] if s else ""

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    data = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def parse_system_state_from_text(text: str) -> str:
    if not text:
        return ""
    if "<|SystemIdle|>" in text or "SystemIdle" in text:
        return "SystemIdle"
    if "<|SystemSpeaking|>" in text or "SystemSpeaking" in text:
        return "SystemSpeaking"
    return ""

def parse_assistant_triple(text: str) -> Tuple[str, str, str]:
    """
    兼容两种输出：
    1) 两 token：<|DecisionStrategy|><|SpecificScenario|>
    2) 三 token：<|SystemIdle|><|DecisionStrategy|><|SpecificScenario|>
    """
    if not text:
        return "", "", ""
    tokens = []
    i = 0
    while True:
        a = text.find("<|", i)
        if a < 0:
            break
        b = text.find("|>", a)
        if b < 0:
            break
        tokens.append(normalize_token_tag(text[a:b + 2]))
        i = b + 2

    if len(tokens) >= 3 and tokens[0] in ("SystemIdle", "SystemSpeaking"):
        return tokens[0], tokens[1], tokens[2]
    if len(tokens) >= 2:
        return "", tokens[0], tokens[1]
    return "", "", ""

def infer_gt_from_messages(sample: Dict[str, Any]) -> str:
    """
    1) assistant 取 action/scenario
    2) 最后一句 user 取 SystemState
    3) 合并映射到完整标签
    """
    msgs = sample.get("messages") or []
    last_assistant = ""
    for msg in reversed(msgs):
        if msg.get("role") == "assistant":
            last_assistant = msg.get("content", "")
            break

    _, action, scenario = parse_assistant_triple(last_assistant)

    system_state = ""
    for msg in reversed(msgs):
        if msg.get("role") == "user":
            system_state = parse_system_state_from_text(msg.get("content", ""))
            if system_state:
                break

    key = TRIPLE2LABEL.get((
        normalize_token_tag(system_state),
        normalize_token_tag(action),
        normalize_token_tag(scenario)
    ))
    return normalize_label(key)

def _is_label_like_assistant_turn(msg: Dict[str, Any]) -> bool:
    if msg.get("role") != "assistant":
        return False
    c = str(msg.get("content", ""))
    return ("<|" in c and "|>" in c)

def _sanitize_sample_no_audio_path(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    写日志/错误样本用：移除或替换 audio 字段，避免暴露路径。
    """
    out = json.loads(json.dumps(sample, ensure_ascii=False))  # 深拷贝
    msgs = out.get("messages") or []
    for m in msgs:
        if "audio" in m:
            m["audio"] = "<redacted>"
    return out

def _trim_history_by_turns(history: List[Dict[str, Any]], max_turns: int) -> List[Dict[str, Any]]:
    """
    history 是 last_user 之前的消息序列（user/assistant 混合）
    这里按“轮”截断：从末尾往前数 max_turns 个 user 轮，并保留其后的 assistant。
    """
    if max_turns <= 0:
        return []

    user_idxs = [i for i, m in enumerate(history) if m.get("role") == "user"]
    if not user_idxs:
        return []

    if len(user_idxs) <= max_turns:
        return history

    keep_from_user = user_idxs[-max_turns]
    return history[keep_from_user:]

def extract_history_and_last_user(sample: Dict[str, Any], max_history_turns: int):
    """
    返回：
      - history（不含最后 user；已截断轮数；且防止 label assistant 泄漏）
      - last_user dict
      - system_state
      - last_user_audio_path（仅内部用，不写日志）
    """
    msgs = sample.get("messages") or []
    last_user_idx = None
    for i in reversed(range(len(msgs))):
        if msgs[i].get("role") == "user":
            last_user_idx = i
            break
    if last_user_idx is None:
        return [], {}, "", ""

    history = msgs[:last_user_idx]
    last_user = msgs[last_user_idx]

    while history and _is_label_like_assistant_turn(history[-1]):
        history = history[:-1]

    history = _trim_history_by_turns(history, max_history_turns)

    state = parse_system_state_from_text(last_user.get("content", ""))
    audio_path = str(last_user.get("audio", "") or "").strip()
    return history, last_user, state, audio_path

# ============================================================
# Prompt
# ============================================================
def build_system_prompt() -> str:
    speaking_lines = "\n".join([
        "<|Maintain|><|Backchannel|>",
        "<|Maintain|><|Invalidation|>",
        "<|Maintain|><|SideTalk|>",
        "<|Maintain|><|Distraction|>",
        "<|StopandListen|><|Interruption|>",
        "<|StopandListen|><|Dismissal|>",
        "<|StopandListen|><|Collaboration|>",
    ])
    idle_lines = "\n".join([
        "<|Takeover|><|Completion|>",
        "<|Takeover|><|Cooperation|>",
        "<|Dismiss|><|Incomplete|>",
        "<|Dismiss|><|Invalidation|>",
        "<|Dismiss|><|Dismissal|>",
        "<|Dismiss|><|Exclusion|>",
        "<|Dismiss|><|SideTalk|>",
    ])

    explain = "\n".join([
        "当 SystemState = SystemSpeaking：",
        "- <|Maintain|><|Backchannel|>：短促附和/跟随，信息量极低，不提新需求，不要求停。",
        "- <|Maintain|><|Invalidation|>：无效声学信号（咳嗽/噪声/不可辨识），不构成对话内容。",
        "- <|Maintain|><|SideTalk|>：对第三方讲话，语用对象不是系统。",
        "- <|Maintain|><|Distraction|>：环境音/广播等与话题无关，应忽略。",
        "- <|StopandListen|><|Interruption|>：明确插话/纠正/提新需求，想让系统立刻停下并听。",
        "- <|StopandListen|><|Dismissal|>：明确要求系统停止/结束互动。",
        "- <|StopandListen|><|Collaboration|>：第三方介入且与任务相关，系统应停下听。",
        "",
        "当 SystemState = SystemIdle：",
        "- <|Takeover|><|Completion|>：用户话轮完成（EOU），句子完整、意图清晰。",
        "- <|Takeover|><|Cooperation|>：第三方提供对任务有价值的协作信息。",
        "- <|Dismiss|><|Incomplete|>：用户没说完/思考中/半句停住/自我修正（那个…/我想…/等下…），系统不抢答继续等。",
        "- <|Dismiss|><|Invalidation|>：无效声学信号（咳嗽/噪声/不可辨识），继续等待。",
        "- <|Dismiss|><|Dismissal|>：用户撤销/终止需求（算了/不用了/结束）。",
        "- <|Dismiss|><|Exclusion|>：语用排除系统参与（不是跟你说的/你别管）。",
        "- <|Dismiss|><|SideTalk|>：对第三方说话但未显式排除系统，不要求系统响应。",
    ])

    return f"""
你是一个【严格的轮次控制 / 对话状态分类器】（turn-taking decision model）。
输入包含【history】与【last_user（纯文本）】。
你的任务：只对【最后一句用户输入】打标签，但必须利用【history】来消歧。

========================
【核心强制规则】SystemState 是已知且固定的
========================
最后一句 user 文本中会明确标注系统当前状态：
- 系统当前状态: <|SystemSpeaking|> 或 <|SystemIdle|>
✅ 你必须基于这个明确给出的 SystemState 进行分类
✅ 你不需要预测 SystemState，只需要输出对应的后两个标签 (DecisionStrategy, SpecificScenario)
✅ 分类结果必须严格匹配该 SystemState 对应的标签列表，不能跨状态输出

========================
你必须结合上下文（强约束）
========================
- 结合上一轮系统/用户内容判断是否抢话/插话
- 结合对话主题判断相关协作输入 vs 无关干扰
- 结合说话对象判断对系统 vs 对第三方

提示：短词“对/嗯/好”通常是附和，但若出现纠正/提问/改需求，则可能是 Interruption 开头。

========================
标签定义（解释）
========================
{explain}

========================
输出格式（严格闭集：两token）
========================
当 SystemState = SystemSpeaking 时，只能输出以下之一：
{speaking_lines}

当 SystemState = SystemIdle 时，只能输出以下之一：
{idle_lines}

硬性规则：
1) 只输出候选字符串，不要解释
2) 不要换行，不要加空格，不要加标点
""".strip()

# ============================================================
# 音频读取：优先 torchaudio；无则 fallback soundfile
# 输出：np.ndarray float32, 16k, mono
# ============================================================
def load_audio_16k_mono(audio_path: str) -> np.ndarray:
    try:
        import torchaudio
        wav, sr = torchaudio.load(audio_path)  # (C, T)
        if wav.numel() == 0:
            return np.zeros((0,), dtype=np.float32)
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        wav = wav.squeeze(0)  # (T,)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        return wav.contiguous().cpu().numpy().astype(np.float32)
    except Exception:
        import soundfile as sf
        y, sr = sf.read(audio_path, dtype="float32", always_2d=False)
        if y is None:
            return np.zeros((0,), dtype=np.float32)
        if y.ndim == 2:
            y = y.mean(axis=1)
        if sr != 16000:
            from scipy.signal import resample_poly
            g = np.gcd(sr, 16000)
            up = 16000 // g
            down = sr // g
            y = resample_poly(y, up, down).astype(np.float32)
        return y.astype(np.float32)

# ============================================================
# MiniCPM 本地推理（user 音频输入，assistant 文本输出）
# ============================================================
class MiniCPMTurnTakingClassifier:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.system_prompt = build_system_prompt()
        self._gen_lock = threading.Lock()

        self.model = AutoModel.from_pretrained(
            model_dir,
            trust_remote_code=True,
            attn_implementation="sdpa",
            torch_dtype=torch.bfloat16,
            init_vision=False,
            init_audio=True,
            init_tts=False,
        )
        self.model.eval()

        if torch.cuda.is_available():
            self.model.cuda()

    def _build_chat_messages(
        self,
        history: List[Dict[str, Any]],
        state: str,
        last_user_audio_path: str,
    ) -> List[Dict[str, Any]]:
        """
        你要的范式：
          - history：user 喂音频 waveform；assistant 喂文本（忽略 assistant 的 audio）
          - last user：必须保留 SystemState 文本 + 音频 waveform（满足硬规则）
        """
        msgs: List[Dict[str, Any]] = []

        # 1) system
        msgs.append({"role": "system", "content": [self.system_prompt]})

        # 2) history：user 音频 / assistant 文本
        for m in history:
            role = m.get("role")
            if role not in ("user", "assistant"):
                continue

            if USER_AUDIO_ASSISTANT_TEXT_HISTORY:
                if role == "user":
                    audio_path = str(m.get("audio", "") or "").strip()
                    if audio_path:
                        audio_wave = load_audio_16k_mono(audio_path)
                        msgs.append({"role": "user", "content": [audio_wave]})
                    else:
                        if not SKIP_HISTORY_WITHOUT_AUDIO:
                            msgs.append({"role": "user", "content": [""]})
                    continue

                if role == "assistant":
                    txt = str(m.get("content", "") or "").strip()
                    msgs.append({"role": "assistant", "content": [txt]})
                    continue

            # fallback：如果你想回退到纯文本 history（这里保持安全兜底）
            txt = str(m.get("content", "") or "").strip()
            msgs.append({"role": role, "content": [txt]})

        # 3) last user：状态文本 + 音频 waveform（满足硬规则）
        state_text = f"系统当前状态: <|{state}|>"

        if not last_user_audio_path:
            msgs.append({"role": "user", "content": [state_text]})
            return msgs

        audio_wave = load_audio_16k_mono(last_user_audio_path)
        msgs.append({"role": "user", "content": [state_text, audio_wave]})
        return msgs

    def predict(self, sample: Dict[str, Any]) -> Optional[str]:
        history, _last_user, state, audio_path = extract_history_and_last_user(
            sample, max_history_turns=MAX_HISTORY_TURNS
        )
        if state not in ("SystemIdle", "SystemSpeaking"):
            return None

        try:
            msgs = self._build_chat_messages(history, state, audio_path)

            with self._gen_lock:
                res = self.model.chat(
                    msgs=msgs,
                    do_sample=DO_SAMPLE,
                    temperature=TEMPERATURE,
                    max_new_tokens=MAX_NEW_TOKENS,
                    use_tts_template=False,
                    generate_audio=False,  # ✅ assistant 输出文本
                )

            if isinstance(res, dict):
                text = str(res.get("text", "") or res.get("response", "") or "").strip()
            else:
                text = str(res).strip()

            if not text:
                return None

            _, action, scenario = parse_assistant_triple(text)
            pred_key = TRIPLE2LABEL.get((
                state,
                normalize_token_tag(action),
                normalize_token_tag(scenario)
            ))
            return pred_key if pred_key in LABELS else None

        except Exception as e:
            print(f"[预测错误] {type(e).__name__}: {e}")
            return None

# ============================================================
# 评测模块（保存错误样本 + 输出 md 报告）——保持你的结构，只加“脱敏”
# ============================================================
@dataclass
class ConfusionMatrix:
    labels: List[str]
    mat: np.ndarray = field(init=False)
    idx: Dict[str, int] = field(init=False)
    total: int = 0
    correct: int = 0

    def __post_init__(self):
        self.idx = {lbl: i for i, lbl in enumerate(self.labels)}
        self.mat = np.zeros((len(self.labels), len(self.labels)), dtype=int)

    def update(self, pred, gt):
        if pred not in self.idx or gt not in self.idx:
            return
        self.mat[self.idx[gt], self.idx[pred]] += 1
        self.total += 1
        if pred == gt:
            self.correct += 1

    def overall(self):
        return self.correct / self.total if self.total else None

    def per_class(self):
        res = {}
        for lbl, i in self.idx.items():
            s = self.mat[i].sum()
            res[lbl] = self.mat[i, i] / s if s else None
        return res

def fmt(x):
    return f"{x * 100:.2f}%" if x is not None else "N/A"

def write_markdown_report(out_dir: Path, cm: ConfusionMatrix, skipped: int, log_path: Path):
    per_class_raw = cm.per_class()

    lines = []
    lines.append("# Benchmark Report")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- Overall Acc: **{fmt(cm.overall())}**")
    lines.append(f"- Evaluated Samples: **{cm.total}**")
    lines.append(f"- Skipped/Fail Samples: **{skipped}**")
    lines.append(f"- Per-sample log: `{log_path.as_posix()}`")
    lines.append("")

    lines.append("## Per-class Accuracy")
    lines.append("")
    lines.append("| Label | Acc | SystemState | DecisionStrategy | SpecificScenario | 说明 |")
    lines.append("|---|---:|---|---|---|---|")

    for k in LABELS:
        acc = fmt(per_class_raw.get(k))
        ss, ds, sc = LABEL2TRIPLE[k]
        zh = (LABEL_META.get(k, {}) or {}).get("zh", "")
        zh = zh.replace("\n", " ").replace("|", "\\|")
        lines.append(f"| {k} | {acc} | {ss} | {ds} | {sc} | {zh} |")

    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path

def run_benchmark(model: MiniCPMTurnTakingClassifier, datasets: List[Path], out_dir: Path, num_workers: int = 16):
    out_dir.mkdir(exist_ok=True, parents=True)
    cm = ConfusionMatrix(LABELS)
    skipped = 0

    logs: List[Dict[str, Any]] = []
    error_samples: List[Dict[str, Any]] = []

    def eval_one(file_path: str, idx: int, sample: Dict[str, Any]) -> Dict[str, Any]:
        gt = infer_gt_from_messages(sample)
        if gt not in LABELS:
            return {"file": file_path, "idx": idx, "status": "skip", "gt": gt}

        t0 = time.time()
        pred = model.predict(sample)
        cost = round(time.time() - t0, 3)

        if pred not in LABELS:
            return {
                "file": file_path,
                "idx": idx,
                "status": "fail",
                "gt": gt,
                "pred": pred,
                "time": cost,
                "sample": _sanitize_sample_no_audio_path(sample),
            }

        correct = (pred == gt)
        if not correct:
            return {
                "file": file_path,
                "idx": idx,
                "status": "wrong",
                "gt": gt,
                "pred": pred,
                "correct": False,
                "time": cost,
                "sample": _sanitize_sample_no_audio_path(sample),
            }

        return {
            "file": file_path,
            "idx": idx,
            "status": "ok",
            "gt": gt,
            "pred": pred,
            "correct": True,
            "time": cost,
        }

    futures = []
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        for path in datasets:
            if not path.exists():
                print(f"[缺失] {path}")
                continue

            samples = read_jsonl(path)
            print(f"[数据] {path}：{len(samples)} 条")
            for i, s in enumerate(samples):
                futures.append(ex.submit(eval_one, str(path), i, s))

        for fu in as_completed(futures):
            r = fu.result()
            st = r.get("status")

            if st == "skip":
                skipped += 1
                continue

            if st == "fail":
                skipped += 1
                logs.append({k: v for k, v in r.items() if k != "sample"})
                error_samples.append(r)
                continue

            if st == "wrong":
                gt = r["gt"]
                pred = r["pred"]
                cm.update(pred, gt)
                logs.append({k: v for k, v in r.items() if k != "sample"})
                error_samples.append(r)
                continue

            gt = r["gt"]
            pred = r["pred"]
            cm.update(pred, gt)
            logs.append(r)

    log_path = out_dir / "per_sample.jsonl"
    with log_path.open("w", encoding="utf-8") as f:
        for l in logs:
            f.write(json.dumps(l, ensure_ascii=False) + "\n")

    err_path = out_dir / "error_samples.jsonl"
    with err_path.open("w", encoding="utf-8") as f:
        for e in error_samples:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    report_path = write_markdown_report(out_dir, cm, skipped, log_path)

    per_class_raw = cm.per_class()
    per_class_report = {}
    for k in LABELS:
        per_class_report[k] = {
            "acc": fmt(per_class_raw.get(k)),
            "explain_zh": LABEL_META.get(k, {}).get("zh", ""),
            "triple": {
                "SystemState": LABEL2TRIPLE[k][0],
                "DecisionStrategy": LABEL2TRIPLE[k][1],
                "SpecificScenario": LABEL2TRIPLE[k][2],
            }
        }

    res = {
        "overall_acc": fmt(cm.overall()),
        "evaluated_samples": cm.total,
        "skipped_or_error": skipped,
        "per_sample_log": str(log_path),
        "error_samples_log": str(err_path),
        "report_md": str(report_path),
        "per_class": per_class_report,
        "prompt_used": model.system_prompt,
        "num_workers": num_workers,
        "max_history_turns": MAX_HISTORY_TURNS,
        "user_audio_assistant_text_history": USER_AUDIO_ASSISTANT_TEXT_HISTORY,
        "skip_history_without_audio": SKIP_HISTORY_WITHOUT_AUDIO,
        "local_model_dir": "<redacted>",
        "max_new_tokens": MAX_NEW_TOKENS,
    }

    print("\n===== 最终结果 =====")
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return res

# ============================================================
# 主函数,如果想测试英文数据集的话可以直接替换即可
# ============================================================
def main():
    input_files = [
        "./Benchmark_Datasets/jsonls/ZH/syn/SystemIdle_Dismiss_Dismissal.jsonl",
        "./Benchmark_Datasets/jsonls/ZH/syn/SystemIdle_Dismiss_Exclusion.jsonl",
        "./Benchmark_Datasets/jsonls/ZH/syn/SystemIdle_Dismiss_Incomplete.jsonl",
        "./Benchmark_Datasets/jsonls/ZH/syn/SystemIdle_Dismiss_Invalidation.jsonl",
        "./Benchmark_Datasets/jsonls/ZH/syn/SystemIdle_Dismiss_SideTalk.jsonl",
        "./Benchmark_Datasets/jsonls/ZH/syn/SystemIdle_Takeover_Completion.jsonl",
        "./Benchmark_Datasets/jsonls/ZH/syn/SystemIdle_Takeover_Cooperation.jsonl",
        "./Benchmark_Datasets/jsonls/ZH/syn/SystemSpeaking_Maintain_Backchannel.jsonl",
        "./Benchmark_Datasets/jsonls/ZH/syn/SystemSpeaking_Maintain_Distraction.jsonl",
        "./Benchmark_Datasets/jsonls/ZH/syn/SystemSpeaking_Maintain_Invalidation.jsonl",
        "./Benchmark_Datasets/jsonls/ZH/syn/SystemSpeaking_Maintain_SideTalk.jsonl",
        "./Benchmark_Datasets/jsonls/ZH/syn/SystemSpeaking_StopandListen_Collaboration.jsonl",
        "./Benchmark_Datasets/jsonls/ZH/syn/SystemSpeaking_StopandListen_Dismissal.jsonl",
        "./Benchmark_Datasets/jsonls/ZH/syn/SystemSpeaking_StopandListen_Interruption.jsonl",
    ]

    datasets = [Path(p) for p in input_files]
    out_dir = Path("./benchmark_minicpm_5")
    out_dir.mkdir(exist_ok=True, parents=True)

    model = MiniCPMTurnTakingClassifier(LOCAL_MODEL_DIR)
    run_benchmark(model, datasets, out_dir, num_workers=NUM_WORKERS)

if __name__ == "__main__":
    main()