import torch
import numpy as np
from PIL import Image

# 颜色转换工具
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    if len(hex_color) != 6:
        raise ValueError(f"无效的十六进制颜色格式: {hex_color}。应为6位字符。")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

# 主色块替换逻辑，单张图片
def replace_multiple_color_blocks_pil(img, source_hex_colors, target_hex_colors, tolerance=10, strategy_on_mismatch="cycle_replacement"):
    if not source_hex_colors or not target_hex_colors:
        return img
    source_rgbs = [hex_to_rgb(h) for h in source_hex_colors]
    target_rgbs = [hex_to_rgb(h) for h in target_hex_colors]
    arr = np.array(img)
    h, w, c = arr.shape
    arr_flat = arr.reshape(-1, 3)
    num_source_colors = len(source_rgbs)
    num_target_colors = len(target_rgbs)
    for i, source_rgb in enumerate(source_rgbs):
        diff = np.abs(arr_flat - source_rgb)
        mask = np.all(diff <= tolerance, axis=1)
        if i < num_target_colors:
            arr_flat[mask] = target_rgbs[i]
        else:
            if strategy_on_mismatch == "cycle_replacement":
                actual_target_index = i % num_target_colors
                arr_flat[mask] = target_rgbs[actual_target_index]
            elif strategy_on_mismatch == "no_replace_excess_source":
                pass
            else:
                pass
    arr = arr_flat.reshape(h, w, 3)
    return Image.fromarray(arr.astype(np.uint8))

class ReplaceColorByPalette:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "输入图片，支持批量。"}),
                "source_hex_colors": ("STRING", {"multiline": True, "tooltip": "源颜色列表，每行一个十六进制色值。"}),
                "target_hex_colors": ("STRING", {"multiline": True, "tooltip": "目标颜色列表，每行一个十六进制色值。"}),
                "color_tolerance": ("INT", {"default": 10, "min": 0, "max": 255, "step": 1, "tooltip": "颜色匹配容差。"}),
                "strategy_on_mismatch": (["cycle_replacement", "no_replace_excess_source"], {"default": "cycle_replacement", "tooltip": "替换策略。"}),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "replace"
    CATEGORY = "image/processing"
    DESCRIPTION = "批量替换图片中的多个指定颜色块为新颜色。"

    def replace(self, image, source_hex_colors, target_hex_colors, color_tolerance, strategy_on_mismatch):
        # 解析颜色列表
        if isinstance(source_hex_colors, str):
            source_hex_colors = [x.strip() for x in source_hex_colors.splitlines() if x.strip()]
        if isinstance(target_hex_colors, str):
            target_hex_colors = [x.strip() for x in target_hex_colors.splitlines() if x.strip()]
        # ComfyUI的IMAGE为torch.Tensor，shape: (B, H, W, 3)，值域0~1
        imgs = []
        for i in range(image.shape[0]):
            img_np = (image[i].cpu().numpy() * 255).clip(0,255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)
            out_img = replace_multiple_color_blocks_pil(pil_img, source_hex_colors, target_hex_colors, color_tolerance, strategy_on_mismatch)
            out_np = np.array(out_img).astype(np.float32) / 255.0
            imgs.append(torch.from_numpy(out_np))
        out_tensor = torch.stack(imgs, dim=0).to(image.device)
        return (out_tensor,)

NODE_CLASS_MAPPINGS = {
    "Replace Color By Palette": ReplaceColorByPalette,
}
