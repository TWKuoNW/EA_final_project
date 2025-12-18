# manual_wm_rollout.py
#
# 功能：
#   1. 載入 DINO-WM 的 PushT world model
#   2. 用 PushT env 的 obs 格式當「模板」，把你給的圖片塞進去
#   3. 定義兩個函式：
#        encoder_fn(frame_path)              -> 現在這張圖的 patch embeddings
#        predictor_fn(frame_path, action_xy) -> 做 action 後，下一步的 patch embeddings
#
# 使用方式（在 main 最下面）：
#
#   frame_path = "your_image.png"   # 換成你自己的圖片
#   action_xy  = (-0.7, -1.3)
#   embeddings = encoder_fn(...)
#   result     = predictor_fn(...)
#
#   print(embeddings.shape)  # (1, num_patches, 384)
#   print(result.shape)      # (1, num_patches, 384)

import os
from pathlib import Path

import numpy as np
import torch
import gym
from PIL import Image
from omegaconf import OmegaConf
import hydra

from preprocessor import Preprocessor
from utils import seed as set_seed

# 關鍵：單純 import，讓 env 相關東西（包含 pusht）註冊進 gym
from env.venv import SubprocVectorEnv  # 不會用到這個 class，只是為了觸發註冊

# ========= 基本設定 =========

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


# ========= 載入 world model =========

def load_ckpt(snapshot_path: Path, device):
    with snapshot_path.open("rb") as f:
        payload = torch.load(f, map_location=device)
    result = {}
    for k, v in payload.items():
        if k in ALL_MODEL_KEYS:
            result[k] = v.to(device)
    result["epoch"] = payload["epoch"]
    print(f"Loaded checkpoint epoch = {result['epoch']}")
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

    print("num_hist       :", wm.num_hist)
    print("encoder emb_dim:", wm.encoder.emb_dim)
    print("total emb_dim  :", wm.emb_dim)
    return wm


# ========= 建 dataset / preprocessor / 單一 env =========

def build_dataset_env_preprocessor(cfg, device):
    # dataset: 用來建 preprocessor
    env_dsets, traj_dsets = hydra.utils.call(
        cfg.env.dataset,
        num_hist=cfg.num_hist,
        num_pred=cfg.num_pred,
        frameskip=cfg.frameskip,
    )
    dset_valid = traj_dsets["valid"]
    print("Loaded", len(env_dsets["train"]), "rollouts")
    print("Loaded", len(dset_valid), "rollouts")

    preprocessor = Preprocessor(
        action_mean  = dset_valid.action_mean,
        action_std   = dset_valid.action_std,
        state_mean   = dset_valid.state_mean,
        state_std    = dset_valid.state_std,
        proprio_mean = dset_valid.proprio_mean,
        proprio_std  = dset_valid.proprio_std,
        transform    = dset_valid.transform,
    )

    # env: 單一環境，不用 SubprocVectorEnv（避免 reset(seed=...) 問題）
    env = gym.make(cfg.env.name, *cfg.env.args, **cfg.env.kwargs)

    return dset_valid, preprocessor, env


# ========= 用 env 的 obs 當「模板」，把圖片塞進去 =========

def make_obs_from_image_using_env(frame_path, env):
    """
    步驟：
      1. 用 env.reset() 拿一個 obs，裡面有 'visual' / 'proprio'
      2. 讀你的圖片，resize 成一樣的 H,W
      3. 把 obs 的 visual 換成你的圖片（保持 layout 不變）
      4. 加 batch=1, time=1 維度，變成 (1,1,...) 格式
    """

    # 兼容舊版 / 新版 gym API
    out = env.reset()
    if isinstance(out, tuple):
        obs, info = out
    else:
        obs = out
        info = {}

    visual  = obs["visual"]
    proprio = obs["proprio"]

    visual_np  = np.array(visual, dtype=np.float32)
    proprio_np = np.array(proprio, dtype=np.float32)

    if visual_np.ndim != 3:
        raise ValueError(f"env obs visual 維度不是 3，拿到 {visual_np.shape}，我不會處理 QQ")

    img = Image.open(frame_path).convert("RGB")

    # 判斷是 HWC 還是 CHW，保持 layout 不變
    if visual_np.shape[-1] == 3:
        # (H, W, C)
        H, W, C = visual_np.shape
        img_resized = img.resize((W, H))
        img_np = np.array(img_resized, dtype=np.float32) / 255.0  # (H,W,3)
        visual_new = img_np
    elif visual_np.shape[0] == 3:
        # (C, H, W)
        C, H, W = visual_np.shape
        img_resized = img.resize((W, H))
        img_np = np.array(img_resized, dtype=np.float32) / 255.0  # (H,W,3)
        visual_new = np.transpose(img_np, (2, 0, 1))              # (3,H,W)
    else:
        raise ValueError(f"env obs visual 既不是 CHW 也不是 HWC，shape={visual_np.shape}")

    # 加 batch=1, time=1
    visual_bt  = visual_new[None, None, ...]   # (1,1,*,*,*)
    proprio_bt = proprio_np[None, None, ...]   # (1,1,proprio_dim)

    obs_0 = {
        "visual":  visual_bt,
        "proprio": proprio_bt,
    }
    return obs_0


# ========= encoder(frame_path)：現在這一張的 latent =========

def encoder_fn(wm, preprocessor, env, frame_path, device):
    """
    encoder(frame):
      - 把圖片塞進 env 模板 → obs_0
      - 用 preprocessor 做跟訓練時一樣的 transform
      - 丟進 wm.encode
      - 拿出 "visual" patch embeddings

    回傳：
      visual_emb: (1, num_patches, 384)
    """
    # 1. 圖片 → obs_0（shape 跟訓練時一樣）
    obs_raw = make_obs_from_image_using_env(frame_path, env)

    # 2. transform（Preprocessor 內部會用 torchvision，回傳 torch.Tensor）
    obs_trans = preprocessor.transform_obs(obs_raw)
    obs_trans = {k: v.to(device) for k, v in obs_trans.items()}

    # 3. 準備「舊 action」（history），先全 0
    act_in_chans = wm.action_encoder.patch_embed.in_channels
    act_hist = torch.zeros(1, 1, act_in_chans, device=device)

    with torch.no_grad():
        z_t = wm.encode(obs_trans, act_hist)      # (1,1,num_patches,emb_dim)
        z_obs, z_act = wm.separate_emb(z_t)
        visual_emb = z_obs["visual"][:, 0]        # (1, num_patches, 384)

    return visual_emb


# ========= predictor(frame_path, action_xy)：下一步 latent =========

def predictor_fn(wm, preprocessor, env, frame_path, action_xy, device):
    """
    predictor(frame, action):
      - 跟 encoder 一樣先算出現在這張圖的 z_t
      - 用 wm.predict 想像未來
      - 用 replace_actions_from_z 把 action_xy 塞進未來 latent
      - 拿出下一步的 visual patch embeddings

    回傳：
      next_visual: (1, num_patches, 384)
    """
    x, y = action_xy

    # 1. 圖片 → obs_0
    obs_raw = make_obs_from_image_using_env(frame_path, env)

    # 2. transform
    obs_trans = preprocessor.transform_obs(obs_raw)
    obs_trans = {k: v.to(device) for k, v in obs_trans.items()}

    act_in_chans = wm.action_encoder.patch_embed.in_channels

    with torch.no_grad():
        # 3. 現在的 z_t
        act_hist = torch.zeros(1, 1, act_in_chans, device=device)
        z_t = wm.encode(obs_trans, act_hist)              # (1,1,P,emb_dim)

        # 4. predictor 想像未來
        z_hist = z_t[:, -wm.num_hist:]                    # 這裡其實就是 z_t 本人
        z_seq  = wm.predict(z_hist)                       # (1,window_len,P,emb_dim)
        z_next = z_seq[:, -1:, ...]                       # (1,1,P,emb_dim)

        # 5. 把你給的 action_xy 塞進 z_next 的 action 部分
        my_action = torch.zeros(1, 1, act_in_chans, device=device)
        if act_in_chans >= 1:
            my_action[0, 0, 0] = x
        if act_in_chans >= 2:
            my_action[0, 0, 1] = y

        z_next = wm.replace_actions_from_z(z_next, my_action)  # (1,1,P,emb_dim)

        # 6. 拆出下一步的 visual patch embeddings
        z_obs_next, z_act_next = wm.separate_emb(z_next)
        next_visual = z_obs_next["visual"][:, 0]          # (1,P,384)

    return next_visual


# ========= main：照你說的 API 走一次 =========

def main():
    set_seed(SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 1. 讀 hydra 設定
    model_path     = f"{CKPT_BASE_PATH}/outputs/{MODEL_NAME}"
    hydra_cfg_path = os.path.join(model_path, "hydra.yaml")
    if not os.path.exists(hydra_cfg_path):
        raise FileNotFoundError(f"找不到 hydra.yaml: {hydra_cfg_path}")
    cfg = OmegaConf.load(hydra_cfg_path)

    # 2. 載 world model
    model_ckpt = Path(model_path) / "checkpoints" / f"model_{MODEL_EPOCH}.pth"
    wm = load_world_model(model_ckpt, cfg, device)

    # 3. 建 dataset / preprocessor / env
    dset_valid, preprocessor, env = build_dataset_env_preprocessor(cfg, device)
    print("action_mean:", dset_valid.action_mean)
    print("action_std :", dset_valid.action_std)

    print("cfg.num_hist:", cfg.num_hist)
    print("cfg.frameskip:", cfg.frameskip)
    print("wm.num_hist:", wm.num_hist)



    # ====== 這裡開始就是你要的使用方式 ======

    frame_path = "img_folder/4.png"
    action_xy  = (0.0, 0.0)

    embeddings = encoder_fn(wm, preprocessor, env, frame_path, device)
    result     = predictor_fn(wm, preprocessor, env, frame_path, action_xy, device)

    print("embeddings shape:", embeddings.shape)  # (1, num_patches, 384)
    print("result shape    :", result.shape)      # (1, num_patches, 384)
    
    save_path_root = "wm_output"
    # 如果你要後面玩 cosine / L2：
    np.save(f"{save_path_root}/embeddings_now.npy", embeddings.cpu().numpy())
    np.save(f"{save_path_root}/embeddings_next.npy", result.cpu().numpy())
    print(f"已存成 {save_path_root}/embeddings_now.npy / {save_path_root}/embeddings_next.npy")


if __name__ == "__main__":
    main()
