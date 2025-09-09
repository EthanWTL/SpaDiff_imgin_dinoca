# test_uv_from_loader.py
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

from loader import build_train_loader  # uses your MazeFlowDataset

# ---- config ----
OUT_DIR = "./scripts/dataloader_exp/_debug/checks"
FRAME_T = 0          # which frame (0..48) to visualize

def save_uv_heatmaps(u: np.ndarray, v: np.ndarray, out_prefix: str):
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    # u heatmap
    plt.figure()
    plt.imshow(u, origin="upper")
    plt.title("u (right +)")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_u.png", dpi=120, bbox_inches="tight", pad_inches=0)
    plt.close()
    # v heatmap
    plt.figure()
    plt.imshow(v, origin="upper")
    plt.title("v (down +)")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_v.png", dpi=120, bbox_inches="tight", pad_inches=0)
    plt.close()

def tensor_to_pil_chw01(img_t: torch.Tensor):
    """img_t: [3,H,W] in [0,1] -> uint8 PIL"""
    arr = (img_t.clamp(0,1) * 255).byte().permute(1,2,0).cpu().numpy()
    return Image.fromarray(arr)

def describe_image(img_t: torch.Tensor, label: str = "image"):
    """
    img_t: [3,H,W] in [0,1]
    Prints stats and returns dict of metrics.
    """
    assert img_t.ndim == 3 and img_t.shape[0] == 3, f"Expected [3,H,W], got {tuple(img_t.shape)}"
    H, W = img_t.shape[1], img_t.shape[2]
    flat = img_t.flatten()
    stats = {
        "shape": (3, H, W),
        "dtype": str(img_t.dtype),
        "min": float(flat.min().item()),
        "max": float(flat.max().item()),
        "mean": float(flat.mean().item()),
        "frac_eq_1": float((flat == 1.0).float().mean().item()),
        "frac_eq_0": float((flat == 0.0).float().mean().item()),
    }
    # fraction of exactly-white pixels across channels
    white_mask = (img_t == 1.0).all(dim=0)  # [H,W]
    stats["white_ratio"] = float(white_mask.float().mean().item())
    print(f"\n--- {label} ---")
    for k, v in stats.items():
        print(f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}")
    return stats

def save_hist(img_t: torch.Tensor, out_path: str):
    vals = img_t.flatten().cpu().numpy()
    if vals.size > 400_000:
        step = max(1, vals.size // 400_000)
        vals = vals[::step]
    plt.figure()
    plt.hist(vals, bins=50)
    plt.title("Pixel value histogram (0..1)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight", pad_inches=0)
    plt.close()

if __name__ == "__main__":
    # Determinism (optional)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.use_deterministic_algorithms(True)

    ds, dl = build_loader(
        batch_size=1,
        num_workers=0,
        shuffle=False,
        # three toggles:
        return_flow_seq=True,     # to get batch['flow']  -> [B,T,2,H,W]
        return_image_bg=True,     # to get batch['image'] -> [B,3,H,W]
        return_hsv_seq=True,      # to get batch['flow_hsv'] -> [B,T,3,H,W]
        return_hsv_channels_seq=True,
        # if you want HSV video under 'image' instead, set hsv_as_image=True in build_loader
    )

    batch = next(iter(dl))  # first batch

    # -----------------------------
    # 1) Flow U/V heatmaps (frame T)
    # -----------------------------
    if "flow" in batch:
        flow = batch["flow"][0]  # [T,2,H,W] (torch)
        assert flow.ndim == 4 and flow.shape[0] >= FRAME_T + 1 and flow.shape[1] == 2, \
            f"Unexpected flow shape: {tuple(flow.shape)}"

        flow_np = flow.detach().cpu().numpy()     # [T,2,H,W]
        u = flow_np[FRAME_T, 0]                   # [H,W]
        v = flow_np[FRAME_T, 1]                   # [H,W]

        print("flow shape:", tuple(flow_np.shape))
        print("u/v shapes:", u.shape, v.shape)
        print("u stats:", float(u.mean()), float(np.max(np.abs(u))))
        print("v stats:", float(v.mean()), float(np.max(np.abs(v))))

        out_prefix = os.path.join(OUT_DIR, f"batch0_frame{FRAME_T:02d}")
        save_uv_heatmaps(u, v, out_prefix)
        print(f"Saved: {out_prefix}_u.png and {out_prefix}_v.png")
    else:
        print("No 'flow' in batch — skip U/V heatmaps.")

    # -----------------------------
    # 2) Background image sanity check
    # -----------------------------
    if "image" in batch and batch["image"].ndim == 4 and batch["image"].shape[1] == 3:
        # sequence present under 'image' (likely HSV video if hsv_as_image=True)
        pass  # handled below in HSV section
    elif "image" in batch:
        img = batch["image"][0]  # [3,H,W]
        describe_image(img, label="background image")
        png_path = os.path.join(OUT_DIR, "image_check_frame0.png")
        tensor_to_pil_chw01(img).save(png_path)
        save_hist(img, os.path.join(OUT_DIR, "image_check_hist.png"))
        print(f"\nSaved:\n  {png_path}\n  {os.path.join(OUT_DIR, 'image_check_hist.png')}")
    else:
        print("No 'image' (background) in batch — skip image checks.")

    # -----------------------------
    # 3) HSV video → GIF (49 frames)
    # -----------------------------
    hsv_video = None
    hsv_src = None
    if "flow_hsv" in batch:
        hsv_video = batch["flow_hsv"][0]   # [T,3,H,W], float 0..1
        hsv_src = "flow_hsv"
    elif "image" in batch and batch["image"].ndim == 4 and batch["image"].shape[1] == 3:
        hsv_video = batch["image"][0]      # [T,3,H,W], float 0..1 (when hsv_as_image=True)
        hsv_src = "image (hsv_as_image=True)"

    if hsv_video is not None:
        T = hsv_video.shape[0]
        frames = []
        for t in range(T):
            frame = (hsv_video[t].clamp(0,1) * 255).byte().permute(1,2,0).cpu().numpy()
            frames.append(Image.fromarray(frame))
        os.makedirs(OUT_DIR, exist_ok=True)
        gif_path = os.path.join(OUT_DIR, "flow_hsv_video.gif")
        frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=80, loop=0)
        print(f"Saved HSV video from {hsv_src}: {gif_path}")
    else:
        print("No HSV video found. Enable return_hsv_seq=True, or set hsv_as_image=True to place HSV under 'image'.")

    if "flow_hsv_channels" in batch:
        hsv_ch = batch["flow_hsv_channels"][0]    # [T,3,H,W] float 0..1
        Hmap, Smap, Vmap = hsv_ch[FRAME_T, 0].cpu().numpy(), hsv_ch[FRAME_T, 1].cpu().numpy(), hsv_ch[FRAME_T, 2].cpu().numpy()

        plt.figure(figsize=(12,3.8))
        for k, (name, mat) in enumerate([('H (angle)', Hmap), ('S (mask)', Smap), ('V (norm mag)', Vmap)]):
            plt.subplot(1,3,k+1)
            plt.imshow(mat, origin='upper', vmin=0, vmax=1)
            plt.title(name); plt.colorbar(fraction=0.046, pad=0.04)
            plt.axis('off')
        os.makedirs(OUT_DIR, exist_ok=True)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"hsv_channels_frame{FRAME_T:02d}.png"), dpi=120)
        plt.close()
        print("Saved HSV channels heatmaps.")
