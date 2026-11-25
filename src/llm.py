"""
Local LLM wrapper for the two compared generators (Qwen3-8B and Gemma 3 12B-IT).

Both models are loaded from Hugging Face in 4-bit NF4 so they fit on a free Colab T4 GPU.
Only one model should be on the GPU at a time: call unload() before loading the other.
"""
import gc
import re
import threading

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          TextIteratorStreamer)

from .config import CONFIG

THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def pick_dtype(name):
    """Choose the compute dtype for one model (see CONFIG["compute_dtype"])."""
    choice = CONFIG["compute_dtype"].get(name, "auto")
    if choice != "auto":
        return getattr(torch, choice)
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        return torch.bfloat16          # Ampere or newer (L4, A100)
    return torch.float16               # T4 (Turing) has no native bfloat16


def gpu_memory_gb():
    if not torch.cuda.is_available():
        return 0.0
    return round(torch.cuda.memory_allocated() / 1024 ** 3, 2)


class LocalLLM:
    """Load one Hugging Face chat model in 4-bit and generate answers from chat messages."""

    def __init__(self, name, model_id=None):
        self.name = name
        self.model_id = model_id or CONFIG["llms"][name]
        dtype = pick_dtype(name)
        self.dtype = str(dtype).replace("torch.", "")
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        # AutoModelForCausalLM maps Gemma 3 to Gemma3ForConditionalGeneration (text input works).
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, quantization_config=quant, dtype=dtype, device_map="auto")
        self.model.eval()
        self.is_qwen3 = "qwen3" in self.model_id.lower()

    # ------------------------------------------------------------ prompt -> tensors
    def _inputs(self, messages):
        extra = {"enable_thinking": False} if self.is_qwen3 else {}   # Qwen3: no reasoning trace
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **extra)
        # The chat template already contains special tokens such as <bos>.
        return self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)

    def _gen_kwargs(self, inputs):
        kwargs = dict(**inputs, max_new_tokens=CONFIG["max_new_tokens"],
                      do_sample=CONFIG["do_sample"],
                      pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        if not CONFIG["do_sample"]:
            # Greedy decoding; clear sampling defaults from generation_config to avoid warnings.
            kwargs.update(temperature=None, top_p=None, top_k=None)
        return kwargs

    # ------------------------------------------------------------ generation
    @torch.inference_mode()
    def generate(self, messages):
        """Return the full answer as one string."""
        inputs = self._inputs(messages)
        output = self.model.generate(**self._gen_kwargs(inputs))
        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return THINK_BLOCK.sub("", text).strip()

    def stream(self, messages):
        """Yield the answer piece by piece (used by the Streamlit app)."""
        inputs = self._inputs(messages)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        kwargs = self._gen_kwargs(inputs)
        kwargs["streamer"] = streamer
        thread = threading.Thread(target=self._generate_no_grad, kwargs=kwargs, daemon=True)
        thread.start()
        for piece in streamer:
            yield piece
        thread.join()

    def _generate_no_grad(self, **kwargs):
        with torch.inference_mode():
            self.model.generate(**kwargs)

    def smoke_test(self):
        """Quick check right after loading. Empty or odd output usually means a dtype problem."""
        reply = self.generate([{"role": "user", "content": "Reply with the word OK only."}])
        ok = len(reply.strip()) > 0 and "ok" in reply.lower()
        print(f"{self.name} ({self.dtype}): {reply!r} -> {'looks fine' if ok else 'CHECK compute_dtype in CONFIG'}")
        return ok

    def unload(self):
        """Remove the model from the GPU."""
        del self.model
        del self.tokenizer
        free_gpu()
