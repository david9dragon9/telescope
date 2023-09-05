from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, TextStreamer
import torch
import copy
import os
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings

str_to_dtype = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def is_llama_model(pretrained_path):
    return "lmsys" in pretrained_path or "llama" in pretrained_path.lower()


class LLM:
    def __init__(
        self,
        pretrained_model_name_or_path,
        context_window=8192,
        max_new_tokens=512,
        precision=torch.bfloat16,
        device_map="auto",
        use_auth_token=None,
    ):
        use_auth_token = (
            os.environ["HF_AUTH_TOKEN"]
            if use_auth_token is None and "HF_AUTH_TOKEN" in os.environ
            else use_auth_token
        )
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=True,
            use_auth_token=use_auth_token,
        )
        if context_window is not None and hasattr(self.config, "max_seq_len"):
            self.config.max_seq_len = context_window
        self.context_window = context_window
        self.config.init_device = "meta"
        if isinstance(precision, str):
            precision = str_to_dtype[precision]
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            config=self.config,
            torch_dtype=precision,
            device_map=device_map,
            trust_remote_code=True,
            use_auth_token=use_auth_token,
        )
        if is_llama_model(pretrained_model_name_or_path):
            print("Initializing using SLOW LlamaTokenizer for Llama models.")
            from transformers.models.llama.tokenization_llama import LlamaTokenizer

            self.tokenizer = LlamaTokenizer.from_pretrained(
                pretrained_model_name_or_path,
                padding_side="left",
                use_auth_token=use_auth_token,
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path,
                padding_side="left",
                trust_remote_code=True,
                use_auth_token=use_auth_token,
            )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.streamer = TextStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        self.generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }

    def decode(self, tokens, skip_special_tokens=True):
        return self.tokenizer.batch_decode(
            tokens, skip_special_tokens=skip_special_tokens
        )

    def tokenize(self, text):
        return self.tokenizer(
            text,
            padding=True,
            max_length=self.context_window,
            truncation=True,
            return_tensors="pt",
        )

    def respond(self, x, stream=True, temperature=1.0):
        generated = self.generate(x, stream=stream, temperature=temperature)
        return self.decode(generated)[0]

    def generate(self, x, stream=False, temperature=1.0):
        if isinstance(x, str):
            tokenized = self.tokenize(x)
        else:
            tokenized = x

        gkwargs = copy.deepcopy(self.generate_kwargs)
        if stream:
            gkwargs["streamer"] = self.streamer
        gkwargs["temperature"] = temperature
        gkwargs["do_sample"] = True
        return self.model.generate(input_ids=tokenized["input_ids"].cuda(), **gkwargs)[
            :, tokenized["input_ids"].shape[1] :
        ]


def get_embedding_llm(model_name):
    return SentenceTransformerEmbeddings(model_name=model_name)
