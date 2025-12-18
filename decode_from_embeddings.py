# decode_from_embeddings.py
#
# 功能（近似版 decoder）：
#   給一個 embeddings.npy （例如 embeddings_next.npy）
#   → 用 PushT 的 dataset + world model encoder
#   → 找出「embedding 最像」的那一張 PushT 圖片
#   → 存成 png
#
# 用法：
#   python decode_from_embeddings.py
#
# 你可以在 main 裡改：
#   save_dir  = "decoded_output"
#   npy_path  = "wm_output/embeddings_next.npy"

import os
from pathlib import Path

import numpy as np
from PIL import Image

import torch
from omegaconf import OmegaConf
import hydra

from preprocessor import Preprocessor
from utils import seed as set_seed

# ======== 基本設定（依照你的環境改） ========

CKPT_BASE_PATH = "/home/kuonw/文件/dino_wm"
MODEL_NAME     = "pusht"
MODEL_EPOCH    = "latest"
SEED           = 0

ALL_MODEL_KEYS = [
    "encoder",
    "predictor",
    "decoder",
    "proprio_encoder",
    "action_encoder",
]


# ======== 載入 world model ========

def load_ckpt(snapshot_path: Path, device):
    with snapshot_path.open("rb") as f:
        payload = torch.load(f, map_location=device)
    result = {}
    for k, v in payload.items():
        if k in ALL_MODEL_KEYS:
            result[k] = v.to(device)
    result["epoch"] = payload["epoch"]
    print(f"[INFO] Loaded checkpoint epoch = {result['epoch']}")
    return result


def load_world_model(model_ckpt: Path, cfg, device):
    if not model_ckpt.exists():
        raise FileNotFoundError(f"找不到模型檔案: {model_ckpt}")

    payload = load_ckpt(model_ckpt, device)

    if "encoder" not in payload:
        payload["encoder"] = hydra.utils.instantiate(cfg.encoder)
    if "predictor" not in payload:
        raise ValueError("Predictor not found in checkpoint")

    # decoder 多半用不到，這裡先照 config 做
    if cfg.has_decoder and "decoder" not in payload:
        if cfg.env.decoder_path is not None:
            base_path    = os.path.dirname(os.path.abspath(__file__))
            decoder_path = os.path.join(base_path, cfg.env.decoder_path)
            ckpt = torch.load(decoder_path)
            if isinstance(ckpt, dict):
                payload["decoder"] = ckpt["decoder"]
            else:
                payload["decoder"] = ckpt
        else:
            payload["decoder"] = None
    elif not cfg.has_decoder:
        payload["decoder"] = None

    wm = hydra.utils.instantiate(
        cfg.model,
        encoder         = payload["encoder"],
        proprio_encoder = payload["proprio_encoder"],
        action_encoder  = payload["action_encoder"],
        predictor       = payload["predictor"],
        decoder         = payload["decoder"],
        proprio_dim     = cfg.proprio_emb_dim,
        action_dim      = cfg.action_emb_dim,
        concat_dim      = cfg.concat_dim,
        num_action_repeat  = cfg.num_action_repeat,
        num_proprio_repeat = cfg.num_proprio_repeat,
    )
    wm.to(device)
    wm.eval()

    print("[INFO] num_hist       :", wm.num_hist)
    print("[INFO] encoder emb_dim:", wm.encoder.emb_dim)
    print("[INFO] total emb_dim  :", wm.emb_dim)
    return wm


# ======== 建 dataset / preprocessor（不用真的啟 env） ========

def build_dataset_and_preprocessor(cfg):
    # env_dsets: state / reward 等等
    # traj_dsets: 觀測序列
    env_dsets, traj_dsets = hydra.utils.call(
        cfg.env.dataset,
        num_hist=cfg.num_hist,
        num_pred=cfg.num_pred,
        frameskip=cfg.frameskip,
    )
    dset_valid = traj_dsets["valid"]
    print("[INFO] Loaded", len(env_dsets["train"]), "env rollouts")
    print("[INFO] Loaded", len(dset_valid), "traj rollouts (valid)")

    preprocessor = Preprocessor(
        action_mean  = dset_valid.action_mean,
        action_std   = dset_valid.action_std,
        state_mean   = dset_valid.state_mean,
        state_std    = dset_valid.state_std,
        proprio_mean = dset_valid.proprio_mean,
        proprio_std  = dset_valid.proprio_std,
        transform    = dset_valid.transform,
    )

    return dset_valid, preprocessor


# ======== 把 patch embeddings 變成「整張圖的一個向量」 ========

def global_embed_from_patches(arr: np.ndarray) -> np.ndarray:
    """
    arr: 可能是 (1, P, D) 或 (P, D) 或 (D,)
    輸出: (D,) 的 1D 向量
    """
    arr = np.asarray(arr)
    print(f"[INFO] 原始 embedding shape = {arr.shape}")
    if arr.ndim == 3:
        # (B,P,D) -> 合併 B,P 再平均
        B, P, D = arr.shape
        arr = arr.reshape(B * P, D)
    elif arr.ndim == 2:
        # (P,D)
        P, D = arr.shape
    elif arr.ndim == 1:
        # (D,) 直接回傳
        return arr.astype(np.float32)

    # 對所有 patch 取平均，得到一個「代表整張圖」的向量
    emb = arr.mean(axis=0).astype(np.float32)
    print(f"[INFO] 轉成 global embedding, shape = {emb.shape}")
    return emb


# ======== 用 world model encoder 把 dataset 中某一張 obs 變成 embedding ========
def encode_single_frame_to_global_emb(wm, preprocessor, obs_visual, obs_proprio, device) -> np.ndarray:
    """
    給 dataset 中的一個 frame：
      obs_visual: (H,W,3) 或 (3,H,W)
      obs_proprio: (proprio_dim,)
    回傳:
      global_emb: (D,) 的 numpy 向量
    """

    # ---- 1. 先把 visual 轉成 HWC 格式 ----
    visual_np = np.array(obs_visual, dtype=np.float32)

    if visual_np.ndim != 3:
        raise ValueError(f"obs_visual 維度不是 3D，拿到 shape={visual_np.shape}")

    if visual_np.shape[-1] == 3:
        # 已經是 (H, W, 3) → 直接用
        visual_hwc = visual_np
    elif visual_np.shape[0] == 3:
        # 是 (3, H, W) → 轉成 (H, W, 3)
        visual_hwc = np.transpose(visual_np, (1, 2, 0))
    else:
        raise ValueError(f"obs_visual 既不是 CHW 也不是 HWC，shape={visual_np.shape}")

    # ---- 2. 加 batch=1, time=1 維度，跟 env 版本一樣 ----
    visual_bt  = visual_hwc[None, None, ...]                      # (1,1,H,W,3)
    proprio_np = np.array(obs_proprio, dtype=np.float32)
    proprio_bt = proprio_np[None, None, ...]                      # (1,1,proprio_dim)

    obs_0 = {
        "visual":  visual_bt,
        "proprio": proprio_bt,
    }

    # ---- 3. 做跟訓練時一樣的 transform（交給 Preprocessor） ----
    obs_trans = preprocessor.transform_obs(obs_0)
    obs_trans = {k: v.to(device) for k, v in obs_trans.items()}

    # ---- 4. 準備 action history（全 0） ----
    act_in_chans = wm.action_encoder.patch_embed.in_channels
    act_hist = torch.zeros(1, 1, act_in_chans, device=device)

    # ---- 5. 丟進 world model encoder，拿出 visual 的 patch embeddings ----
    with torch.no_grad():
        z_t = wm.encode(obs_trans, act_hist)      # (1,1,P,emb_dim)
        z_obs, z_act = wm.separate_emb(z_t)
        visual_emb = z_obs["visual"][:, 0]        # (1,P,384) for visual part

    # ---- 6. 對所有 patch 取平均 → 一個 global 向量 (D,) ----
    visual_np = visual_emb.cpu().numpy()
    global_emb = global_embed_from_patches(visual_np)  # (D,)
    return global_emb


# ======== 「近似 decoder」：embedding.npy -> 找最像的 PushT 圖 ========

def decoder_from_npy(
    npy_path: str,
    wm,
    preprocessor,
    dset_valid,
    device,
    max_trajs: int = 5,
    max_frames_per_traj: int = 50,
):
    """
    decoder(npy_path):
      1. 讀 embeddings.npy，變成 target_emb (D,)
      2. 在 dset_valid 中掃一些 frame，找出 embedding 最接近的那張圖
      3. 回傳該圖的畫面 (H,W,3) uint8

    注意：
      - max_trajs, max_frames_per_traj 可以調整，越大越精確但越慢。
    """

    # 1. 讀 target embedding
    emb_arr = np.load(npy_path)
    target_emb = global_embed_from_patches(emb_arr)  # (D,)

    best_dist = float("inf")
    best_img  = None

    # 2. 在 valid dataset 裡面找
    num_traj = min(len(dset_valid), max_trajs)
    print(f"[INFO] 在前 {num_traj} 條軌跡中搜尋，每條最多 {max_frames_per_traj} 個 frame")

    for traj_idx in range(num_traj):
        obs, act, state, env_info = dset_valid[traj_idx]
        visuals  = obs["visual"]   # shape: (T, ...)
        proprios = obs["proprio"]  # shape: (T, proprio_dim)
        T = visuals.shape[0]

        frames_to_check = min(T, max_frames_per_traj)
        for t in range(frames_to_check):
            obs_visual  = visuals[t]
            obs_proprio = proprios[t]

            # 2-1. 這一張圖的 embedding
            cand_emb = encode_single_frame_to_global_emb(
                wm, preprocessor, obs_visual, obs_proprio, device
            )  # (D,)

            # 2-2. 算 L2 distance
            diff = cand_emb - target_emb
            dist = float(np.sqrt((diff * diff).sum()))

            if dist < best_dist:
                best_dist = dist
                # 存下原始畫面（轉成 HWC, uint8）
                img_np = np.array(obs_visual, dtype=np.float32)

                if img_np.ndim == 3 and img_np.shape[0] == 3:
                    # CHW -> HWC
                    img_np = np.transpose(img_np, (1, 2, 0))

                # 嘗試標準化到 [0,255]
                if img_np.max() <= 1.0:
                    img_np = img_np * 255.0

                img_np = np.clip(img_np, 0, 255).astype(np.uint8)
                best_img = img_np

        print(f"[INFO] 已檢查 traj {traj_idx+1}/{num_traj}, 目前最小距離 = {best_dist:.4f}")

    if best_img is None:
        raise RuntimeError("沒有成功找到任何候選圖片 (best_img is None)")

    print(f"[INFO] 搜尋結束，最小 L2 距離 = {best_dist:.4f}")
    return best_img


# ======== main：照你想要的 API 方式呼叫 ========

def main():
    set_seed(SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("[INFO] Using device:", device)

    # 1. 讀 hydra 設定
    model_path     = f"{CKPT_BASE_PATH}/outputs/{MODEL_NAME}"
    hydra_cfg_path = os.path.join(model_path, "hydra.yaml")
    if not os.path.exists(hydra_cfg_path):
        raise FileNotFoundError(f"找不到 hydra.yaml: {hydra_cfg_path}")
    cfg = OmegaConf.load(hydra_cfg_path)

    # 2. 載 world model
    model_ckpt = Path(model_path) / "checkpoints" / f"model_{MODEL_EPOCH}.pth"
    wm = load_world_model(model_ckpt, cfg, device)

    # 3. 建 dataset + preprocessor
    dset_valid, preprocessor = build_dataset_and_preprocessor(cfg)

    # ===== 這裡開始照你想像的架構 =====
    save_dir = "decoded_output"
    os.makedirs(save_dir, exist_ok=True)

    npy_path = "wm_output/embeddings_next.npy"   # 你要解碼的 embeddings.npy

    # 定義一個只吃 npy_path 的 decoder（把其他東西關在外面）
    def decoder(npy_file: str):
        return decoder_from_npy(
            npy_file,
            wm,
            preprocessor,
            dset_valid,
            device,
            max_trajs=5,            # 可以改大
            max_frames_per_traj=50  # 可以改大
        )

    # 實際呼叫
    img = decoder(npy_path)

    # 存檔
    out_path = os.path.join(save_dir, "decoded_from_embeddings.png")
    Image.fromarray(img).save(out_path)
    print(f"[INFO] 已將近似解碼結果存成: {out_path}")


if __name__ == "__main__":
    main()
