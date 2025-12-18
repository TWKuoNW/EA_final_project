#!/usr/bin/env python3
"""
predict_from_npy.py

功能：
- 載入 DINO-WM 的 PushT world model
- 讀取一個 visual embedding 檔 (例如 manual_wm_rollout 存的 embeddings_now.npy)
- 把這個 latent 當作「現在的狀態」
- 丟進世界模型的 predictor，算出「下一步的 latent」
- 拆出下一步的 visual patch embeddings，存成 .npy

注意：
- 這裡的 .npy 須是「world model 自己 encoder 出來的 visual latent」
  也就是形狀大致為 (1, P, 384)，而不是 DINOv2 那個 1369 patches 的版本。
"""

import os
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
import hydra

from utils import seed as set_seed

# 只是為了觸發 env 註冊（pusht 等），雖然這支程式不會真的用到 env
from env.venv import SubprocVectorEnv  # noqa: F401

# ========= 基本設定 =========

CKPT_BASE_PATH = "/home/kuonw/文件/dino_wm"
MODEL_NAME     = "pusht"
MODEL_EPOCH    = "latest"
SEED           = 0

# 你要讀的 latent 檔案（建議就是 manual_wm_rollout 存的那個）
EMB_PATH_IN    = "wm_output/embeddings_now.npy"
# 輸出的檔名
EMB_PATH_OUT   = "wm_output/embeddings_next_from_npy.npy"

# 指定一個 action（目前先用簡單的 (x, y)）
ACTION_XY      = (0.0, 0.0)   # 你可以改成 (-0.7, -1.3) 之類


ALL_MODEL_KEYS = [
    "encoder",
    "predictor",
    "decoder",
    "proprio_encoder",
    "action_encoder",
]


# ========= 載入 world model =========

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

    # decoder 可能不在 ckpt 裡，簡化處理
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


# ========= 從 .npy 建出完整 z_t =========

def load_visual_from_npy(path: Path, device: torch.device, encoder_dim: int):
    """
    讀取 embeddings_now.npy：
      - 支援形狀：
          (1, P, D) 或 (P, D)
      - 將其變成 shape = (1, 1, P, D)
    """
    if not path.exists():
        raise FileNotFoundError(f"找不到 latent 檔案: {path}")

    arr = np.load(path).astype(np.float32)
    print(f"[INFO] 讀取 {path}, shape = {arr.shape}")

    if arr.ndim == 3:
        # (B, P, D)
        if arr.shape[0] != 1:
            raise ValueError(f"[ERROR] 預期 batch=1，但拿到 shape={arr.shape}")
        B, P, D = arr.shape
    elif arr.ndim == 2:
        # (P, D)
        P, D = arr.shape
        B = 1
        arr = arr[None, ...]  # (1, P, D)
    else:
        raise ValueError(f"[ERROR] 不支援的 latent shape: {arr.shape}")

    if D != encoder_dim:
        raise ValueError(f"[ERROR] latent 維度 D={D} 和 encoder_dim={encoder_dim} 不一致")

    visual = torch.from_numpy(arr).to(device)      # (1, P, D)
    visual = visual.unsqueeze(1)                   # (1, 1, P, D)

    print(f"[INFO] 轉成 visual_t shape = {visual.shape}")
    return visual  # (1,1,P,D)


def pack_full_z_from_visual(visual_t: torch.Tensor, wm):
    """
    把只有 visual 的 latent（(B, T, P, Dv)）塞到完整 latent z_t 裡：
      - Dv = encoder_dim
      - full_dim = wm.emb_dim
      - 前 Dv 維放 visual，後面補 0（proprio + action）
    """
    B, T, P, Dv = visual_t.shape
    full_dim = wm.emb_dim

    if full_dim < Dv:
        raise ValueError(f"[ERROR] full_dim({full_dim}) < visual_dim({Dv})")

    z_full = torch.zeros(B, T, P, full_dim, device=visual_t.device)
    z_full[..., :Dv] = visual_t

    print(f"[INFO] 建立完整 z_t，shape = {z_full.shape}")
    return z_full


def apply_action_to_z_next(wm, z_next: torch.Tensor, action_xy):
    """
    將你指定的 action_xy 塞進 z_next 的 action 部分：

    - z_next: (B, T, P, emb_dim)，通常 B=1, T=1
    - action_xy: (x, y)
    - wm.action_encoder.patch_embed.in_channels 給出 action 向量維度（例如 10）
    """
    x, y = action_xy
    B, T, P, D = z_next.shape

    act_in_chans = wm.action_encoder.patch_embed.in_channels
    my_action = torch.zeros(B, T, act_in_chans, device=z_next.device)

    if act_in_chans >= 1:
        my_action[..., 0] = x
    if act_in_chans >= 2:
        my_action[..., 1] = y

    # 讓 wm 用這個 action 重新寫入 z 的 action 部分
    z_next_updated = wm.replace_actions_from_z(z_next, my_action)

    return z_next_updated


# ========= 主流程：從 .npy 做一步預測 =========

def main():
    set_seed(SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("[INFO] Using device:", device)

    # 1) 讀 hydra 設定
    model_path     = f"{CKPT_BASE_PATH}/outputs/{MODEL_NAME}"
    hydra_cfg_path = os.path.join(model_path, "hydra.yaml")
    if not os.path.exists(hydra_cfg_path):
        raise FileNotFoundError(f"找不到 hydra.yaml: {hydra_cfg_path}")
    cfg = OmegaConf.load(hydra_cfg_path)

    # 2) 載 world model
    model_ckpt = Path(model_path) / "checkpoints" / f"model_{MODEL_EPOCH}.pth"
    wm = load_world_model(model_ckpt, cfg, device)

    # 3) 從 .npy 讀取「現在的 visual latent」
    emb_path_in = Path(EMB_PATH_IN)
    visual_t = load_visual_from_npy(emb_path_in, device, wm.encoder.emb_dim)  # (1,1,P,Dv)

    # 4) 塞到完整 z_t（補 proprio + action = 0）
    z_t = pack_full_z_from_visual(visual_t, wm)   # (1,1,P,emb_dim)

    with torch.no_grad():
        # 5) predictor 想像未來
        #    這裡 z_t 的時間長度是 1，wm.num_hist 可能是 3，但切 -num_hist 仍然會拿到整段（也就是 1）
        z_hist = z_t[:, -wm.num_hist:]            # (1,1,P,emb_dim)
        z_seq  = wm.predict(z_hist)               # (1,window_len,P,emb_dim)
        z_next = z_seq[:, -1:, ...]               # 取最後一個時間步 → (1,1,P,emb_dim)

        # 6) 把你指定的 action_xy 寫進 z_next 的 action 部分
        z_next = apply_action_to_z_next(wm, z_next, ACTION_XY)  # (1,1,P,emb_dim)

        # 7) 拆出下一步的 visual patch embeddings
        z_obs_next, z_act_next = wm.separate_emb(z_next)
        next_visual = z_obs_next["visual"][:, 0]                 # (1,P,Dv)

    # 8) 存檔
    emb_path_out = Path(EMB_PATH_OUT)
    emb_path_out.parent.mkdir(parents=True, exist_ok=True)
    np.save(emb_path_out, next_visual.cpu().numpy())

    print("[INFO] next_visual shape:", next_visual.shape)
    print(f"[INFO] 已存成: {emb_path_out}")


if __name__ == "__main__":
    main()
