import torch
from PIL import Image
import open_clip

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, _, preprocess = open_clip.create_model_and_transforms('MobileCLIP-S1') # pretrained='checkpoints/mobileclip_s2.pt'
tokenizer = open_clip.get_tokenizer('MobileCLIP-S1')

image = preprocess(Image.open("image.png").convert('RGB')).unsqueeze(0)
text = tokenizer(["a diagram", "a dog", "a cat"]).to(device)

model.to(device)
model.eval()

with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
    image_encoder_ms = []
    text_encoder_ms = []

    image = image.half().to(device)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    for _ in range(500):
        
        start.record()
        image_features = model.encode_image(image)
        end.record()
        
        torch.cuda.synchronize()
        image_encoder_ms.append( start.elapsed_time(end))

        start.record()
        text_features = model.encode_text(text)
        end.record()
        
        torch.cuda.synchronize()
        text_encoder_ms.append(start.elapsed_time(end))

    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    # image_features /= image_features.norm(dim=-1, keepdim=True)
    # text_features /= text_features.norm(dim=-1, keepdim=True)

    # text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    print("Image encoder latency:", sum(image_encoder_ms) / len(image_encoder_ms))
    print("Text encoder latency:", sum(text_encoder_ms) / len(text_encoder_ms))

# print("Label probs:", text_probs)