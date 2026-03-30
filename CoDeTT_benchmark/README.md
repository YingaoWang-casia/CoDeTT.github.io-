# CoDeTT Benchmark

CoDeTT Benchmark 用于评测 Turn-Taking（轮次接管）模型在多场景决策任务中的表现，支持统一四分类评测与多模型对比。

## 论文与数据集

- 论文（arXiv）：[CoDeTT: A Context-Aware Decision Benchmark for Turn-Taking Evaluation](https://arxiv.org/abs/2603.25434)
- 数据集（Hugging Face）：[YingaoWang-casia/CoDeTT](https://huggingface.co/datasets/YingaoWang-casia/CoDeTT)
- 数据集（ModelScope）：[wyawya/CoDeTT](https://www.modelscope.cn/datasets/wyawya/CoDeTT)

## 仓库内容

```text
.
├── benchmark.py                         # 主评测入口（统一流程）
├── benchmark_qwen3.py                   # Qwen3-Omni 接口评测脚本
├── benchmark_minicpm.py                 # MiniCPM 本地评测脚本
├── benchmark_ke_semantic.py             # KE-SemanticVAD 评测脚本
├── four_class.py                        # 四分类统计工具
└── requirements.txt
```

## 环境准备

建议 Python 3.10+。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

如果要测试大模型版本，建议单独准备对应运行环境（依赖与显存需求通常不同）。

## 快速开始

### 1) 主评测（推荐）

```bash
python benchmark.py --out_dir ./outputs --run_name exp1
```

说明：
- `benchmark.py` 会读取脚本内默认数据集路径（`DEFAULT_DATASETS_EN` / `DEFAULT_DATASETS_ZH`）。
- 若你本地路径不同，请先修改这些默认路径，以及 `build_default_paths()` 的模型路径。

### 2) Qwen3-Omni 评测

```bash
python benchmark_qwen3.py
```

说明：
- 默认从脚本内 `input_files` 读取数据，输出到 `./qwen3`。
- 可通过环境变量覆盖 API：
  - `QWEN_API_URL`（默认 `http://localhost:8900/v1/chat/completions`）
  - `QWEN_TIMEOUT`
  - `SEND_MODEL_FIELD`

### 3) MiniCPM 评测

```bash
python benchmark_minicpm.py
```

说明：
- 默认从脚本内 `input_files` 读取数据，输出到 `./benchmark_minicpm_5`。
- `LOCAL_MODEL_DIR` 需要指向你本地可用模型目录。

### 4) KE-SemanticVAD 评测

```bash
python benchmark_ke_semantic.py \
  --out_dir ./outputs_ke \
  --history_rounds 0 \
  --datasets_user /path/to/user_1.jsonl /path/to/user_2.jsonl \
  --datasets_agent /path/to/agent_1.jsonl /path/to/agent_2.jsonl
```

说明：
- `--datasets_user` / `--datasets_agent` 不传时，会使用脚本内默认路径。
- `--history_rounds` 语义：`0=仅当前句`，`1=当前句+往上2句`，`2=当前句+往上4句`，`<0=全历史`。

### 5) 标签过滤工具

```bash
python scripts/filter_test_hard_labels.py \
  --input ./datasets/test_hard.jsonl \
  --output ./datasets/test_hard.filtered.jsonl \
  --keep 完整 不完整 附和 Dismissal \
  --add-label-field
```

支持保留标签别名：
- 中文：`完整` / `不完整` / `附和` / `Dismissal`
- 英文：`completion` / `incomplete` / `backchannel` / `dismissal_speaking`

## 输出结果

不同脚本输出目录略有差异，但通常包含：
- `results.json`：总体与分项指标
- `report.md`：Markdown 报告
- `per_sample*.jsonl`：逐样本日志
- `error_samples.jsonl`：失败样本（如脚本有该输出）

## 常见注意事项

1. 当前代码中部分默认路径是硬编码路径（历史环境路径）；在新机器上需要手动改成本地路径。
2. 若出现 “No dataset files found”，优先检查脚本内默认数据路径或通过 CLI 传入数据集路径。
3. 运行 API 相关脚本前，先确认模型服务地址、端口与鉴权配置可用。

## 引用

如果你使用了本仓库或 CoDeTT 数据，建议引用论文：

- [https://arxiv.org/abs/2603.25434](https://arxiv.org/abs/2603.25434)

