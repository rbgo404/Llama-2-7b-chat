from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
import os
from pathlib import Path


class InferlessPythonModel:

    def initialize(self):
        self.repo_id = "meta-llama/Llama-2-7b-chat-hf"  # Specify the model repository ID
        self.HF_TOKEN = "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"  # Access Hugging Face token from environment variable
        self.VOLUME_NFS = os.getenv("VOLUME_NFS")  # Define model storage location

        # Construct model directory path
        self.model_dir = f"{self.VOLUME_NFS}/{self.repo_id}"
        self.model_dir_path = Path(self.model_dir)

        # Create the model directory if it doesn't exist
        if not self.model_dir_path.exists():
            self.model_dir_path.mkdir(exist_ok=True, parents=True)

            # Download the model snapshot from Hugging Face Hub (excluding specific file types)
            snapshot_download(
                self.repo_id,
                local_dir=self.model_dir,
                token=self.HF_TOKEN,  # Provide token if necessary
                ignore_patterns=["*.pt", "*.gguf"],
            )

        # Define sampling parameters for model generation
        # You can set max_tokens to 1024 for complete answer to your question
        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.1,
            repetition_penalty=1.18,
            top_k=40,
            max_tokens=512,
        )

        # Initialize the LLM object with the downloaded model directory
        self.llm = LLM(model=self.model_dir)

        # Load the tokenizer associated with the pre-trained model
        self.tokenizer = AutoTokenizer.from_pretrained(self.repo_id, token=self.HF_TOKEN)

    def infer(self, inputs):
        prompts = inputs["prompt"]  # Extract the prompt from the input

        # Apply the chat template and convert to a list of strings (without tokenization)
        input_text = self.tokenizer.apply_chat_template([{"role": "user", "content": prompts}], tokenize=False)

        # Generate text using the LLM with the specified sampling parameters
        result = self.llm.generate(input_text, self.sampling_params)

        # Extract the generated text from the result object
        result_output = [output.outputs[0].text for output in result]

        # Return a dictionary containing the generated text
        return {"generated_result": result_output[0]}

    def finalize(self):
        self.llm = None
        self.tokenizer = None
