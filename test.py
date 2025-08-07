# Load model directly
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
model_path = "/root/shared-storage/Qwen-7B"  # 模型所在目录
tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype="auto")
count = 0
with open('reasoning_3sample.txt', 'r', encoding='utf-8') as f:
    for line in f:
        content = line.strip()
        # print(content)
        inputs = tokenizer(content, return_tensors="pt").to(model.device)
        input_ids = inputs["input_ids"]
        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss
            perplexity = torch.exp(loss)
        count = count + 1
        print(f"Sentence {count}, Perplexity: {perplexity.item()}")