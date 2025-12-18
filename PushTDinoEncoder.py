# PushTDinoEncoder.py
#
# Minimal class version of manual_wm_rollout.py (encoder only)
# - input: image (path / PIL / RGB np)
# - output: visual patch embeddings (1, num_patches, 384) typically
# - no predictor, no env needed

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf
import hydra

from preprocessor import Preprocessor
from utils import seed as set_seed


ALL_MODEL_KEYS = [
    "encoder",
    "predictor",
    "decoder",
    "proprio_encoder",
    "action_encoder",
]


def _load_ckpt(snapshot_path: Path, device: torch.device) -> Dict[str, Any]:
    if not snapshot_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {snapshot_path}")

    with snapshot_path.open("rb") as f:
        payload = torch.load(f, map_location=device)

    result: Dict[str, Any] = {}
    for k, v in payload.items():
        if k in ALL_MODEL_KEYS and v is not None:
            result[k] = v.to(device)
    result["epoch"] = payload.get("epoch", "unknown")
    return result


def _to_pil_rgb(x: Union[str, Image.Image, np.ndarray]) -> Image.Image:
    if isinstance(x, Image.Image):
        return x.convert("RGB")
    if isinstance(x, str):
        return Image.open(x).convert("RGB")

    arr = np.asarray(x)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"rgb numpy must be (H,W,3). got {arr.shape}")

    if arr.dtype != np.uint8:
        a = arr.astype(np.float32)
        if a.max() <= 1.5:
            a = np.clip(a * 255.0, 0, 255).astype(np.uint8)
        else:
            a = np.clip(a, 0, 255).astype(np.uint8)
        arr = a

    return Image.fromarray(arr, mode="RGB")


class PushTDinoEncoder:
    """
    Encoder-only wrapper (no predictor).
    Equivalent intent to your manual_wm_rollout encoder_fn, but packed as a class
    and does NOT require any gym env.
    """

    def __init__(
        self,
        ckpt_base_path: str = "/home/kuonw/文件/dino_wm",
        model_name: str = "pusht",
        model_epoch: str = "latest",
        seed: int = 0,
        device: Optional[str] = None,
        verbose: bool = True,
    ):
        set_seed(seed)
        self.verbose = verbose

        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # 1) load hydra cfg
        model_dir = Path(ckpt_base_path) / "outputs" / model_name
        hydra_cfg_path = model_dir / "hydra.yaml"
        if not hydra_cfg_path.exists():
            raise FileNotFoundError(f"hydra.yaml not found: {hydra_cfg_path}")
        self.cfg = OmegaConf.load(str(hydra_cfg_path))

        # 2) load ckpt
        ckpt_path = model_dir / "checkpoints" / f"model_{model_epoch}.pth"
        payload = _load_ckpt(ckpt_path, self.device)
        if self.verbose:
            print(f"Loaded checkpoint epoch = {payload.get('epoch')}")

        # 3) build dataset + preprocessor (same as your script)
        env_dsets, traj_dsets = hydra.utils.call(
            self.cfg.env.dataset,
            num_hist=self.cfg.num_hist,
            num_pred=self.cfg.num_pred,
            frameskip=self.cfg.frameskip,
        )
        dset_valid = traj_dsets["valid"]
        if self.verbose:
            print("Loaded", len(env_dsets["train"]), "rollouts")
            print("Loaded", len(dset_valid), "rollouts")

        self.preprocessor = Preprocessor(
            action_mean=dset_valid.action_mean,
            action_std=dset_valid.action_std,
            state_mean=dset_valid.state_mean,
            state_std=dset_valid.state_std,
            proprio_mean=dset_valid.proprio_mean,
            proprio_std=dset_valid.proprio_std,
            transform=dset_valid.transform,
        )

        # 4) instantiate world model (we keep ctor compatible; predictor may exist but unused)
        if "encoder" not in payload:
            payload["encoder"] = hydra.utils.instantiate(self.cfg.encoder).to(self.device)
        if "proprio_encoder" not in payload:
            payload["proprio_encoder"] = hydra.utils.instantiate(self.cfg.proprio_encoder).to(self.device)
        if "action_encoder" not in payload:
            payload["action_encoder"] = hydra.utils.instantiate(self.cfg.action_encoder).to(self.device)

        predictor = payload.get("predictor", None)  # not used, but ctor may require
        decoder = payload.get("decoder", None)

        try:
            self.wm = hydra.utils.instantiate(
                self.cfg.model,
                encoder=payload["encoder"],
                proprio_encoder=payload["proprio_encoder"],
                action_encoder=payload["action_encoder"],
                predictor=predictor,
                decoder=decoder,
                proprio_dim=self.cfg.proprio_emb_dim,
                action_dim=self.cfg.action_emb_dim,
                concat_dim=self.cfg.concat_dim,
                num_action_repeat=self.cfg.num_action_repeat,
                num_proprio_repeat=self.cfg.num_proprio_repeat,
            ).to(self.device)
        except TypeError:
            self.wm = hydra.utils.instantiate(
                self.cfg.model,
                encoder=payload["encoder"],
                proprio_encoder=payload["proprio_encoder"],
                action_encoder=payload["action_encoder"],
                proprio_dim=self.cfg.proprio_emb_dim,
                action_dim=self.cfg.action_emb_dim,
                concat_dim=self.cfg.concat_dim,
                num_action_repeat=self.cfg.num_action_repeat,
                num_proprio_repeat=self.cfg.num_proprio_repeat,
            ).to(self.device)

        self.wm.eval()

        # 5) shapes for wm.encode input
        self.act_in_chans = int(self.wm.action_encoder.patch_embed.in_channels)
        self.proprio_in_chans = int(self.wm.proprio_encoder.patch_embed.in_channels)

        # proprio normalization tensors (match training style)
        pm = np.array(self.preprocessor.proprio_mean, dtype=np.float32).reshape(-1)
        ps = np.array(self.preprocessor.proprio_std, dtype=np.float32).reshape(-1)

        # be tolerant to length mismatch
        if pm.shape[0] != self.proprio_in_chans:
            if pm.shape[0] > self.proprio_in_chans:
                pm = pm[: self.proprio_in_chans]
                ps = ps[: self.proprio_in_chans]
            else:
                pad = self.proprio_in_chans - pm.shape[0]
                pm = np.pad(pm, (0, pad), mode="constant", constant_values=0.0)
                ps = np.pad(ps, (0, pad), mode="constant", constant_values=1.0)

        self._pm = torch.tensor(pm, device=self.device).view(1, 1, -1)
        self._ps = torch.tensor(ps, device=self.device).view(1, 1, -1).clamp(min=1e-6)

        if self.verbose:
            try:
                print("Model emb_dim: ", self.wm.emb_dim)
            except Exception:
                pass
            print("proprio_in_chans:", self.proprio_in_chans)
            print("act_in_chans    :", self.act_in_chans)

    def close(self):
        # no env to close; keep for API symmetry
        pass

    @torch.no_grad()
    def encode(self, img: Union[str, Image.Image, np.ndarray]) -> torch.Tensor:
        pil = _to_pil_rgb(img)

        # Try transform on PIL first; if transform expects Tensor, convert and retry.
        try:
            v = self.preprocessor.transform(pil)
        except TypeError as e:
            msg = str(e)
            if "Tensor Image" in msg or "PIL" in msg or "PIL.Image" in msg:
                # Convert PIL -> Tensor (C,H,W), float in [0,1]
                try:
                    import torchvision.transforms.functional as TF
                    t = TF.pil_to_tensor(pil).float() / 255.0
                except Exception:
                    arr = np.array(pil, dtype=np.float32) / 255.0  # (H,W,3)
                    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # (C,H,W)

                v = self.preprocessor.transform(t)
            else:
                raise

        if not torch.is_tensor(v):
            v = torch.as_tensor(v)

        # v should be (C,H,W) or (1,C,H,W)
        if v.ndim == 4 and v.shape[0] == 1:
            v = v[0]
        if v.ndim != 3:
            raise ValueError(f"visual after transform must be (C,H,W). got {tuple(v.shape)}")

        visual_bt = v.unsqueeze(0).unsqueeze(0).to(self.device)  # (1,1,C,H,W)

        # dummy proprio (zeros) + normalize
        proprio_bt = torch.zeros((1, 1, self.proprio_in_chans), device=self.device)
        proprio_bt = (proprio_bt - self._pm) / self._ps

        obs_trans = {"visual": visual_bt, "proprio": proprio_bt}

        # dummy action history
        act_hist = torch.zeros((1, 1, self.act_in_chans), device=self.device)

        z_t = self.wm.encode(obs_trans, act_hist)      # (1,1,P,emb_dim)
        z_obs, _z_act = self.wm.separate_emb(z_t)
        visual_emb = z_obs["visual"][:, 0]             # (1,P,384)
        return visual_emb



if __name__ == "__main__":
    enc = PushTDinoEncoder(
        ckpt_base_path="/home/kuonw/文件/dino_wm",
        model_name="pusht",
        model_epoch="latest",
        device="cuda:0",  # or "cpu"
        verbose=True,
    )

    emb = enc.encode("img_folder/4.png")
    print("embedding shape:", emb.shape)
