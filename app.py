from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
import os
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class InferlessPythonModel:

    def initialize(self):
        model_id="facebook/opt-125m"  # Specify the model repository ID
        HF_TOKEN = "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" # Access Hugging Face token from environment variable
        VOLUME_NFS = "/var/nfs-mount/llama-2-vol"  # Define model storage location

        # Construct model directory path
        model_dir = f"{VOLUME_NFS}/{model_id}"
        model_dir_path = Path(model_dir)

        # Create the model directory if it doesn't exist
        if not model_dir_path.exists():
            model_dir_path.mkdir(exist_ok=True, parents=True)

            # Download the model snapshot from Hugging Face Hub (excluding specific file types)
            snapshot_download(
                model_id,
                local_dir=model_dir,
                token=HF_TOKEN,  # Provide token if necessary
                ignore_patterns=["*.pt", "*.gguf"],
            )

        # Define sampling parameters for model generation
        # You can set max_tokens to 1024 for complete answer to your question
        
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float16).cuda()
        
        # the fast tokenizer currently does not work correctly
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)


    def infer(self, inputs):
        prompt = inputs["prompt"]  # Extract the prompt from the input

        # from transformers import AutoModelForCausalLM, AutoTokenizer
        # import torch
        
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.cuda()
        
        generated_ids = self.model.generate(input_ids)
        
        result = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return {"generated_result": result}

    def finalize(self):
        self.model = None
        self.tokenizer = None
