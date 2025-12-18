#!/usr/bin/env python3
"""
compare_embeddings.py

功能：
- 讀兩個 .npy 檔
- 把裡面的 embedding 轉成 1D 向量
- 計算：
    - Cosine similarity（越接近 1 越像）
    - L2 distance（越接近 0 越像）

用法：
- 打開這個檔案，修改 main() 裡的 p1, p2 路徑
- 然後在終端機執行：
    python compare_embeddings.py
"""

from pathlib import Path
import numpy as np


# ----------------- 讀檔 + 轉成 1D 向量 -----------------

def load_embedding(path: Path) -> np.ndarray:
    """
    讀取一個 .npy 檔，並轉成 1D 向量 (D,) 當作全局 embedding。

    支援形狀：
      - (1, P, D) 或 (B, P, D)：對 P 取平均 → (B, D)，若 B=1 再壓成 (D,)
      - (P, D)：對 P 取平均 → (D,)
      - (D,)：直接使用
    """
    arr = np.load(path)
    print(f"[INFO] 讀取 {path}, 原始 shape = {arr.shape}")

    arr = arr.astype(np.float32)

    if arr.ndim == 3:
        # (B, P, D) → 對 P 取平均 → (B, D)
        pooled = arr.mean(axis=1)
        if pooled.shape[0] == 1:
            pooled = pooled[0]  # (D,)

    elif arr.ndim == 2:
        # (P, D) → 對 P 取平均 → (D,)
        pooled = arr.mean(axis=0)

    elif arr.ndim == 1:
        pooled = arr

    else:
        raise ValueError(f"[ERROR] 不支援的 embedding shape: {arr.shape}")

    if pooled.ndim != 1:
        raise ValueError(f"[ERROR] 期望最後是 1D 向量，但拿到 shape={pooled.shape}")

    print(f"[INFO] 轉成全局 embedding，shape = {pooled.shape}")
    return pooled


# ----------------- 比較函式 -----------------

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity，越接近 1 越像。"""
    a = a.ravel()
    b = b.ravel()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    """L2 distance，越接近 0 越像。"""
    diff = a.ravel() - b.ravel()
    return float(np.linalg.norm(diff))


# ----------------- 主程式 -----------------

def main():
    # 你在這裡填入檔名就好
    # 可以放相對路徑或絕對路徑
    p1 = "wm_output/4.npy"
    p2 = "wm_output/3.npy"

    path1 = Path(p1)
    path2 = Path(p2)

    if not path1.exists():
        print(f"[ERROR] 找不到檔案: {path1}")
        return
    if not path2.exists():
        print(f"[ERROR] 找不到檔案: {path2}")
        return

    emb1 = load_embedding(path1)
    emb2 = load_embedding(path2)
    print(f"emb1.shape, emb2.shape = {emb1.shape}, {emb2.shape}")


    cos = cosine_similarity(emb1, emb2)
    l2  = l2_distance(emb1, emb2)

    print("\n========== 比較結果 ==========")
    print(f"File 1: {path1.name}")
    print(f"File 2: {path2.name}")
    print(f"  Cosine similarity = {cos:.6f}  （越接近 1 越像）")
    print(f"  L2 distance       = {l2:.6f}  （越接近 0 越像）")


if __name__ == "__main__":
    main()
