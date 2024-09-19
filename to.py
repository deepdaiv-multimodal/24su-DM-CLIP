from transformers import CLIPTokenizer

tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')
test_texts = [["a sample text", "another sample text"]]

try:
    encoded_texts = tokenizer(test_texts, padding=True, truncation=True, return_tensors="pt")
    print(encoded_texts)
except Exception as e:
    print(f"Error during tokenization: {e}")
