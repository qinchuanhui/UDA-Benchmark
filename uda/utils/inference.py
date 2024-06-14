import argparse
from concurrent import futures
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

import grpc

import json
import time


class LLM(object):

    def __init__(self, model_name):
        self.model_name = model_name
        if "llama" in model_name:
            self.model_class = "llama"
        else:
            self.model_class = "others"

    def init_llm(self):
        print(
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "====== Init LLM =======",
        )
        # my_cache_dir = "/data/hf_cache/hub"
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=True,
            trust_remote_code=True,
        )
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.1,
            top_p=0.9,
            repetition_penalty=1.1,
        )
        print(
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "====== LLM Service Started =======",
        )

    def infer(self, messages):
        # print the timedate
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "CallLLM")
        prompt = self.pipe.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if self.model_class == "llama":
            terminators = [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            ]
            output = self.pipe(prompt, eos_token_id=terminators)
        else:
            output = self.pipe(prompt)
        res = output[0]["generated_text"][len(prompt) :]
        return res


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct"
    )
    args = argparser.parse_args()
    model_name = args.model_name
    llm = LLM(model_name)
    llm.init_llm()
