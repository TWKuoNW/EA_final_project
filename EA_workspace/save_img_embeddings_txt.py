import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
import numpy as np

# -------------------------
# 0. 選設備（CPU 或 GPU）
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# 1. 載入 DINOv2 模型
# -------------------------
print("Loading DINOv2 backbone ...")
dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
dinov2.eval().to(device)

# -------------------------
# 2. 影像前處理
# -------------------------
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


def get_img_embedding(image_path: str) -> torch.Tensor:
    """
    給一張圖片路徑，回傳 img_embedding，shape = [384]
    """
    img = Image.open(image_path).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)  # [1, 3, H, W]

    with torch.no_grad():
        features_dict = dinov2.forward_features(x)
        patch_tokens = features_dict["x_norm_patchtokens"]  # [1, N_patches, 384]
        img_embedding = patch_tokens.mean(dim=1)            # [1, 384]

    # squeeze(0) 變成 [384]，比較好存成 txt
    return img_embedding.cpu().squeeze(0)


def save_embedding_txt(embedding: torch.Tensor, filename: str):
    """
    把一個 [384] 的向量存成 txt 檔，每一列一個數字
    """
    np_emb = embedding.numpy()
    np.savetxt(filename, np_emb, fmt="%.6f")
    print(f"Saved: {filename}, shape = {np_emb.shape}")


if __name__ == "__main__":
    # TODO: 這三個檔名換成你實際的圖片檔案
    img_pusht_1 = "img_folder/pushT_img1.png"
    img_pusht_2 = "img_folder/pushT_img2.png"
    img_ele     = "img_folder/elephant.png"

    emb_p1 = get_img_embedding(img_pusht_1)
    emb_p2 = get_img_embedding(img_pusht_2)
    emb_ele = get_img_embedding(img_ele)

    save_embedding_txt(emb_p1, "pusht1_embedding.txt")
    save_embedding_txt(emb_p2, "pusht2_embedding.txt")
    save_embedding_txt(emb_ele, "elephant_embedding.txt")

