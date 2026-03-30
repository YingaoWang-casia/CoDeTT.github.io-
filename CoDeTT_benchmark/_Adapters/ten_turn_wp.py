# /kpfs-data/yingao.wang/code/Turn_Benchamrk/_Adapters/ten_turn_wp.py
from __future__ import annotations

import re
from typing import Any, Dict, Optional

from .base import BaseTurnModel


class TENTurnWP(BaseTurnModel):
    """
    TEN-Turn: 三分类 {complete,incomplete,backchannel}
    """
    model_name = "TEN-Turn"
    supports_audio = False
    supports_text = True
    supports_context = True
    supported_labels = {"complete", "incomplete", "backchannel", "dismissal"}  # ✅ 现支持 dismissal

    def __init__(self, model_path: str, device: Optional[str] = None):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

        self._torch = torch
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        if self.device == "cuda":
            try:
                torch.backends.cuda.enable_flash_sdp(False)
                torch.backends.cuda.enable_mem_efficient_sdp(False)
                torch.backends.cuda.enable_math_sdp(True)
            except Exception:
                pass

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            attn_implementation="eager",
        ).to(self.device).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        pad_id = self.tokenizer.eos_token_id or self.tokenizer.pad_token_id
        self.gen_cfg = GenerationConfig(
            do_sample=False,
            max_new_tokens=4,
            pad_token_id=pad_id,
        )

    def _normalize(self, s: str) -> str:
        s = (s or "").strip().lower()
        if not s:
            return "unknown"

        # ✅ “wait/dismiss/dissmiss …” 统归 dismissal
        if re.search(r"\bwait\b", s) or re.search(r"\bdismiss\b", s) or re.search(r"\bdissmiss\b", s):
            return "dismissal"

        # 先处理否定表达
        if re.search(r"\bnot\s+finished\b", s) or re.search(r"\bnot\s+complete\b", s):
            return "incomplete"

        # ✅ incomplete 判定
        if re.search(r"\bunfinished\b", s) or re.search(r"\bincomplete\b", s):
            return "incomplete"

        if re.search(r"\bfinished\b", s) or re.search(r"\bcomplete\b", s):
            return "complete"

        return "unknown"

    def _make_prompt(self, sample: Dict[str, Any]) -> str:
        ctx = sample.get("context") or []
        if hasattr(self.tokenizer, "apply_chat_template") and ctx:
            messages = [{"role": "system", "content": ""}] + ctx
            return self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        return sample.get("last_text", "")

    def predict(self, sample: Dict[str, Any]) -> Optional[str]:
        text = (sample.get("last_text") or "").strip()
        if not text:
            return None

        prompt = self._make_prompt(sample)
        enc = self.tokenizer(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        with self._torch.inference_mode():
            out = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=self.gen_cfg,
            )

        resp_ids = out[0][input_ids.shape[-1]:]
        resp = self.tokenizer.decode(resp_ids, skip_special_tokens=True)
        lab = self._normalize(resp)

        return None if lab == "unknown" else lab