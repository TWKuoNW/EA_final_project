import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode


# -------------------------
# 0. 選設備（CPU 或 GPU）
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# 1. 載入 DINOv2 模型
#    （這就是 dino_wm 裡面拿來做 visual encoder 的那類模型）
# -------------------------
print("Loading DINOv2 backbone ...")
dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
dinov2.eval().to(device)

# -------------------------
# 2. 定義影像前處理（resize + crop + normalize）
#    518x518 是 DINOv2 預訓練時常用的解析度
# -------------------------
preprocess = T.Compose(
    [
        T.Resize(518, interpolation=InterpolationMode.BICUBIC),
        T.CenterCrop(518),
        T.ToTensor(),
        # ImageNet 常用的 mean / std
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


# -------------------------
# 3. 把「一張圖」轉成 embedding
# -------------------------
def image_to_embeddings(image_path: str):
    """
    輸入：image_path (一張 RGB 照片)
    輸出：
      - patch_embeddings: [1, N_patches, feat_dim]，N_patches 是多少個 patch
      - img_embedding:    [1, feat_dim]，把所有 patch 平均後的一個向量
    """

    # 3-1. 讀圖，轉成 RGB
    img = Image.open(image_path).convert("RGB")

    # 3-2. 做前處理，變成 tensor，shape = [1, 3, H, W]
    x = preprocess(img).unsqueeze(0).to(device)

    # 3-3. 丟進 DINOv2，拿到 patch features
    with torch.no_grad():
        features_dict = dinov2.forward_features(x)
        # 官方做法：從 dict 裡拿 'x_norm_patchtokens'
        # shape: [batch, num_patches, feat_dim]，feat_dim 對 vits14 來說通常是 384
        patch_tokens = features_dict["x_norm_patchtokens"]  # [1, N_patches, 384]

        # 3-4. 如果你只想要「一個向量代表整張圖」：
        img_embedding = patch_tokens.mean(dim=1)  # [1, 384]

    # 全部搬回 CPU，比較好存檔 / 後處理
    return patch_tokens.cpu(), img_embedding.cpu()


# -------------------------
# 4. 小測試：讀一張圖，印出 embedding 形狀
# -------------------------
if __name__ == "__main__":
    test_img = "img_folder/goal.png"

    patches, img_emb = image_to_embeddings(test_img)
    print("patch_embeddings shape:", patches.shape)  # 例：torch.Size([1, 256, 384])
    print("img_embedding shape:", img_emb.shape)      # torch.Size([1, 384])

    # 也可以存成 .npy，之後 EA / dino_wm 可以直接讀
    import numpy as np

    np.save("goal.npy", patches.numpy())
    # np.save("img_embedding.npy", img_emb.numpy())
    print("Saved to patch_embeddings.npy / img_embedding.npy")
