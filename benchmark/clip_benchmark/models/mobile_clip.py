from open_clip import create_model_and_transforms, get_tokenizer
import torch
# import mobileclip

def load_mobile_clip(model_name: str = "mobileclip_s0", pretrained: str = None, cache_dir: str = None, image_encoder_id: str = None, device="cpu"):

    print(f"모델 이름: {model_name}")
    print(f"Pretrained 가중치: {pretrained}")
    print(f"Cache 디렉토리: {cache_dir}")
    print(f"사용 장치: {device}")
    try:
        # model, _, preprocess_val = mobileclip.create_model_and_transforms('mobileclip_s1', pretrained=pretrained)
        # tokenizer = mobileclip.get_tokenizer('mobileclip_s1')
        model, _, preprocess_val = create_model_and_transforms(
            model_name=model_name,
            precision="amp_bfloat16",
            device=device,
            output_dict=True,
            image_encoder_id=image_encoder_id,
        )

        tokenizer = get_tokenizer(model_name)
        
        if pretrained:
            print(f"Pretrained 가중치를 불러옵니다: {pretrained}")
            model.load_state_dict(torch.load(pretrained, map_location=device)["state_dict"])

        model = model.to(device=device, dtype=torch.bfloat16).eval()

        return model, preprocess_val, tokenizer
    except Exception as e:
        print(f"MobileCLIP 모델 로딩 중 오류 발생: {str(e)}")
        raise