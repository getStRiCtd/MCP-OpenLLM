from transformers import AutoModel, AutoTokenizer

MODEL_ID = "Qwen/Qwen2-0.5B-Instruct"



class QwenModel:
    def __init__(self):
        self.model = AutoModel.from_pretrained(MODEL_ID)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    def invoke(self, query):
        pass

