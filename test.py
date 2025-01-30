#%%
from transformers import CLIPTokenizer, CLIPTextModel
import torch

model_name = "openai/clip-vit-large-patch14"
tokenizer = CLIPTokenizer.from_pretrained(model_name)
text_encoder = CLIPTextModel.from_pretrained(model_name)

def encode_text(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        embeddings = text_encoder(**inputs).last_hidden_state[:, 0, :]
    return embeddings  # Shape: (b, d)

texts = ["a cat", "a dog"]
embeddings = encode_text(texts)
print(embeddings.shape)  # Expected: (2, 512) for CLIP-ViT-B/32

# %%
