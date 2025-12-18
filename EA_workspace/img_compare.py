import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading DINOv2 backbone ...")
dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
dinov2.eval().to(device)

preprocess = T.Compose(
    [
        T.Resize(518, interpolation=InterpolationMode.BICUBIC),
        T.CenterCrop(518),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


def image_to_embeddings(image_path: str):
    img = Image.open(image_path).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        features_dict = dinov2.forward_features(x)
        patch_tokens = features_dict["x_norm_patchtokens"]  # [1, N_patches, 384]
        img_embedding = patch_tokens.mean(dim=1)            # [1, 384]

    return patch_tokens.cpu(), img_embedding.cpu()


# -------- 新增：拿「單一圖片」的 img_embedding，順便做 L2 正規化 --------
def get_img_embedding(image_path: str) -> torch.Tensor:
    _, img_emb = image_to_embeddings(image_path)  # [1, 384]
    # 做 L2 正規化，之後算 cosine 比較方便
    img_emb = img_emb / img_emb.norm(dim=1, keepdim=True)
    return img_emb  # shape: [1, 384]


# -------- 新增：兩個向量的 cosine / L2 --------
def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a * b).sum().item())


def l2_dist(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm().item())


if __name__ == "__main__":
    # TODO: 把下面三個檔名換成你實際的檔案
    img_pusht_1   = "img_folder/pushT_img1.png"
    img_pusht_2   = "img_folder/pushT_img2.png"
    img_elephant  = "img_folder/elephant.png"

    e1 = get_img_embedding(img_pusht_1)
    e2 = get_img_embedding(img_pusht_2)
    e_ele = get_img_embedding(img_elephant)

    print("cosine(PushT1, PushT2)    =", cosine_sim(e1, e2))
    print("cosine(PushT1, Elephant) =", cosine_sim(e1, e_ele))

    print("L2(PushT1, PushT2)       =", l2_dist(e1, e2))
    print("L2(PushT1, Elephant)    =", l2_dist(e1, e_ele))
