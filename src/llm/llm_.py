from typing import Optional, Any

from langchain.llms.base import LLM
from langchain_core.callbacks import CallbackManagerForLLMRun
from transformers import AutoModelForCausalLM, AutoTokenizer

# TODO: set from user code
model_name = "Qwen/Qwen2-0.5B-Instruct"
model_type = "Qwen2.0"
transformer_model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model_type = model_type

class HuggingfaceModel(LLM):
    """
    Class for LLM's that were loaded from transformers library
    Copied from https://qwen.readthedocs.io/en/latest/framework/Langchain.html
    """

    def __init__(self):
        super().__init__()

    def _llm_type(self) -> str:
        return self.model_type

    def _call(
        self,
        prompt: str,
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(transformer_model.device)
        generated_ids = transformer_model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response