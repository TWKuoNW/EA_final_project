import os
import random
from pathlib import Path

import gym
import numpy as np
import torch
from einops import rearrange
from omegaconf import OmegaConf
import hydra

from env.venv import SubprocVectorEnv
from preprocessor import Preprocessor
from planning.evaluator import PlanEvaluator
from utils import seed as set_seed

# 這些 key 用來從 checkpoint 把模型撈出來
ALL_MODEL_KEYS = [
    "encoder",
    "predictor",
    "decoder",
    "proprio_encoder",
    "action_encoder",
]


# ======== 這裡是你要改的「硬編碼設定」區 ========

# 1. ckpt_base_path：你的 outputs 資料夾在哪裡
#    如果你的路徑是 ~/文件/dino_wm/outputs/pusht/...
#    那就設成下面這樣（注意：不要寫到 outputs，那一層就好）
CKPT_BASE_PATH = "/home/kuonw/文件/dino_wm"

# 2. 模型名稱（PushT）
MODEL_NAME = "pusht"

# 3. 要用哪一個 checkpoint 檔名：
#    如果你的檔案是 model_latest.pth → 這裡就用 "latest"
#    如果是 model_final.pth        → 這裡就用 "final"
MODEL_EPOCH = "latest"

# 4. 規劃長度：我們要讓模型「想像幾步」推 T
GOAL_H = 30  # 可以改大改小

# 5. 一次要跑幾條軌跡（batch size），先用 1 就好
N_EVALS = 1

# 6. 隨機種子
SEED = 0

# 7. 要從 dataset 拿什麼當目標（先固定用 dset 的起點和終點）
GOAL_SOURCE = "dset"  # 先不要改，改懂之後再玩


# ======== 輔助函式：讀 checkpoint、組模型 ========

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


def load_model(model_ckpt: Path, train_cfg, num_action_repeat: int, device):
    result = {}
    if model_ckpt.exists():
        result = load_ckpt(model_ckpt, device)
        print(f"Resuming from checkpoint: {model_ckpt}")
    else:
        raise FileNotFoundError(f"找不到模型檔案: {model_ckpt}")

    # encoder / predictor / decoder 等設定方式與原本 plan.py 一樣
    if "encoder" not in result:
        result["encoder"] = hydra.utils.instantiate(train_cfg.encoder)
    if "predictor" not in result:
        raise ValueError("Predictor not found in model checkpoint")

    if train_cfg.has_decoder and "decoder" not in result:
        base_path = os.path.dirname(os.path.abspath(__file__))
        if train_cfg.env.decoder_path is not None:
            decoder_path = os.path.join(base_path, train_cfg.env.decoder_path)
            ckpt = torch.load(decoder_path)
            if isinstance(ckpt, dict):
                result["decoder"] = ckpt["decoder"]
            else:
                result["decoder"] = ckpt
        else:
            raise ValueError("需要 decoder，但 config 裡沒有 decoder_path")
    elif not train_cfg.has_decoder:
        result["decoder"] = None

    model = hydra.utils.instantiate(
        train_cfg.model,
        encoder=result["encoder"],
        proprio_encoder=result["proprio_encoder"],
        action_encoder=result["action_encoder"],
        predictor=result["predictor"],
        decoder=result["decoder"],
        proprio_dim=train_cfg.proprio_emb_dim,
        action_dim=train_cfg.action_emb_dim,
        concat_dim=train_cfg.concat_dim,
        num_action_repeat=num_action_repeat,
        num_proprio_repeat=train_cfg.num_proprio_repeat,
    )
    model.to(device)
    model.eval()
    return model


# ======== 從 dataset 抽一段軌跡，拿來當起點 & 目標 ========

def sample_traj_segment_from_dset(dset, n_evals, frameskip, goal_H):
    """
    從 valid dataset 抽 n_evals 條「夠長的軌跡」片段出來，
    用來決定起點 / 終點 / ground-truth 動作。
    """
    traj_len = frameskip * goal_H + 1

    states = []
    actions = []
    observations = []
    env_info = []

    # 先檢查有沒有任何軌跡夠長
    valid_traj = [
        dset[i][0]["visual"].shape[0]
        for i in range(len(dset))
        if dset[i][0]["visual"].shape[0] >= traj_len
    ]
    if len(valid_traj) == 0:
        raise ValueError("資料集中沒有任何軌跡長度 >= traj_len")

    for _ in range(n_evals):
        max_offset = -1
        # 挑到一條夠長的軌跡
        while max_offset < 0:
            traj_id = random.randint(0, len(dset) - 1)
            obs, act, state, e_info = dset[traj_id]
            max_offset = obs["visual"].shape[0] - traj_len

        state = state.numpy()
        offset = random.randint(0, max_offset)

        obs_seg = {key: arr[offset : offset + traj_len] for key, arr in obs.items()}
        state_seg = state[offset : offset + traj_len]
        act_seg = act[offset : offset + frameskip * goal_H]

        actions.append(act_seg)
        states.append(state_seg)
        observations.append(obs_seg)
        env_info.append(e_info)

    return observations, states, actions, env_info


def prepare_targets_from_dset(dset, env, frameskip, goal_H, n_evals, data_preprocessor, eval_seeds):
    """
    仿照 PlanWorkspace.prepare_targets（只保留 dset case），
    回傳 obs_0, obs_g, state_0, state_g, gt_actions。
    """
    observations, states, actions, env_info = sample_traj_segment_from_dset(
        dset, n_evals, frameskip, goal_H
    )

    # 更新 env 設定
    env.update_env(env_info)

    # init_state: 每條軌跡的開頭 state
    init_state = np.array([x[0] for x in states])

    actions = torch.stack(actions)  # (b, T*frameskip, act_dim)

    # 世界模型用的動作（重新 reshape）
    wm_actions = rearrange(actions, "b (t f) d -> b t (f d)", f=frameskip)

    # denormalize 後給真 env 滾一遍，拿 ground-truth obs
    exec_actions = data_preprocessor.denormalize_actions(actions)
    rollout_obses, rollout_states = env.rollout(
        eval_seeds, init_state, exec_actions.numpy()
    )

    obs_0 = {key: np.expand_dims(arr[:, 0], axis=1) for key, arr in rollout_obses.items()}
    obs_g = {key: np.expand_dims(arr[:, -1], axis=1) for key, arr in rollout_obses.items()}
    state_0 = init_state
    state_g = rollout_states[:, -1]

    return obs_0, obs_g, state_0, state_g, wm_actions


# ======== 主流程：不用 hydra.main，直接寫在 main() 裡 ========

def main():
    # ---- 基本設定 ----
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(SEED)

    # ---- 讀取訓練時的 hydra.yaml（裡面描述 env / model 結構）----
    model_path = f"{CKPT_BASE_PATH}/outputs/{MODEL_NAME}"
    hydra_cfg_path = os.path.join(model_path, "hydra.yaml")
    if not os.path.exists(hydra_cfg_path):
        raise FileNotFoundError(f"找不到 hydra.yaml: {hydra_cfg_path}")

    model_cfg = OmegaConf.load(hydra_cfg_path)

    # ---- 建 dataset（valid split）----
    # 和 plan.py 相同寫法
    _, dset_all = hydra.utils.call(
        model_cfg.env.dataset,
        num_hist=model_cfg.num_hist,
        num_pred=model_cfg.num_pred,
        frameskip=model_cfg.frameskip,
    )
    dset_valid = dset_all["valid"]

    # ---- 建 env（SubprocVectorEnv）----
    if model_cfg.env.name in ["wall", "deformable_env"]:
        from env.serial_vector_env import SerialVectorEnv
        EnvClass = SerialVectorEnv
    else:
        EnvClass = SubprocVectorEnv

    env = EnvClass(
        [
            lambda: gym.make(
                model_cfg.env.name, *model_cfg.env.args, **model_cfg.env.kwargs
            )
            for _ in range(N_EVALS)
        ]
    )

    # ---- 載入 world model ----
    num_action_repeat = model_cfg.num_action_repeat
    model_ckpt = Path(model_path) / "checkpoints" / f"model_{MODEL_EPOCH}.pth"
    wm = load_model(model_ckpt, model_cfg, num_action_repeat, device=device)

    # ---- 建 preprocessor（負責 normalize / denormalize）----
    data_preprocessor = Preprocessor(
        action_mean=dset_valid.action_mean,
        action_std=dset_valid.action_std,
        state_mean=dset_valid.state_mean,
        state_std=dset_valid.state_std,
        proprio_mean=dset_valid.proprio_mean,
        proprio_std=dset_valid.proprio_std,
        transform=dset_valid.transform,
    )

    # ---- 準備 evaluation 用的起點 / 目標 ----
    eval_seeds = [SEED * n + 1 for n in range(N_EVALS)]

    if GOAL_SOURCE != "dset":
        raise NotImplementedError("這個簡化版只支援 GOAL_SOURCE='dset'，先這樣用就好。")

    obs_0, obs_g, state_0, state_g, gt_actions = prepare_targets_from_dset(
        dset_valid,
        env,
        frameskip=model_cfg.frameskip,
        goal_H=GOAL_H,
        n_evals=N_EVALS,
        data_preprocessor=data_preprocessor,
        eval_seeds=eval_seeds,
    )

    # ---- 建 evaluator（內部會呼叫 world model + 把結果存影片）----
    evaluator = PlanEvaluator(
        obs_0=obs_0,
        obs_g=obs_g,
        state_0=state_0,
        state_g=state_g,
        env=env,
        wm=wm,
        frameskip=model_cfg.frameskip,
        seed=eval_seeds,
        preprocessor=data_preprocessor,
        n_plot_samples=1,
    )

    # ============================================================
    # 這一段就是「你自己設計動作」的地方（最重要）
    # ============================================================

    # action_dim = world model 動作維度 = dset.action_dim * frameskip
    action_dim = dset_valid.action_dim * model_cfg.frameskip

    # 建一個 actions tensor，形狀：[N_EVALS, GOAL_H, action_dim]
    actions = torch.zeros((N_EVALS, GOAL_H, action_dim), device=device)

    # ====== 你可以改這裡來設計自己的動作 ======
    # 簡單範例：假設前兩維是 (dx, dy)，我們讓每一步都 "往右推 0.2"
    # 注意：這裡是在「模型的動作空間」，不是實際座標，但可以先這樣玩。
    # actions[..., 0] = 0.2
    x = -0.7
    y = -1.3


    # ====== 你之後要接 EA：只要把上面這段改成 EA 產生的 actions 即可 ======
    actions[:, :, 0] = x
    actions[:, :, 1] = -(y) 

    # 每條軌跡的長度 = GOAL_H
    action_len = torch.full(
        (N_EVALS,),
        GOAL_H,
        dtype=torch.int64,
    ).cpu()

    # ============================================================
    # 丟進 evaluator，請世界模型「想像」這些動作會發生什麼
    # 並且存一個影片叫 "my_manual_plan"
    # ============================================================

    logs, successes, _, _ = evaluator.eval_actions(
        actions.detach(),
        action_len,
        save_video=True,
        filename="my_manual_plan",
    )

    print("評估結果：")
    for k, v in logs.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
