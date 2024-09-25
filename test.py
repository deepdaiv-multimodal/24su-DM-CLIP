import torch
from PIL import Image
import open_clip
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, _, preprocess = open_clip.create_model_and_transforms('MobileCLIP-S2') # pretrained='checkpoints/mobileclip_s2.pt'
tokenizer = open_clip.get_tokenizer('MobileCLIP-S2')

image = preprocess(Image.open("image.png").convert('RGB')).unsqueeze(0)
text = tokenizer(["a diagram", "a dog", "a cat"]).to(device)

model.to(device)
model.eval()

with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
    image_encoder_fps = []
    text_encoder_fps = []

    image = image.half().to(device)

    for _ in range(100):
        start = time.time()
        image_features = model.encode_image(image)
        image_encoder_fps.append(1 / (time.time() - start))

        start = time.time()
        text_features = model.encode_text(text)
        text_encoder_fps.append(1 / (time.time() - start))

    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    print("Image encoder FPS:", sum(image_encoder_fps) / len(image_encoder_fps))
    print("Text encoder FPS:", sum(text_encoder_fps) / len(text_encoder_fps))

print("Label probs:", text_probs)