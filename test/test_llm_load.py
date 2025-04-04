from src.llm.llm_ import HuggingfaceModel

print("start")

qwen = HuggingfaceModel()
print(qwen.invoke(f"Hello"))
