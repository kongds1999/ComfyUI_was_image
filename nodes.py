import torch
import numpy as np
from PIL import Image
import re

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
    CATEGORY = "comfyui_was_image"
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

class ConvertGrayToImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gray_tensor": ("IMAGE",),
            }
        }

    CATEGORY = "comfyui_was_image"
    RETURN_TYPES = ("IMAGE",)  # 或者你自定义的类型
    RETURN_NAMES = ("image",)
    FUNCTION = "gray_to_image"

    def gray_to_image(self, gray_tensor):
        images = []
        if gray_tensor.dim() == 2:
            gray_tensor = gray_tensor.unsqueeze(0)
        for img in gray_tensor:
            img_np = (img.cpu().numpy() * 255).astype('uint8')
            img_np = np.expand_dims(img_np, axis=-1)  # (H, W, 1)
            img_np = np.repeat(img_np, 3, axis=-1)    # (H, W, 3) 灰度转伪RGB
            img_np = img_np.astype(np.float32) / 255.0
            images.append(torch.from_numpy(img_np))
        out_tensor = torch.stack(images, dim=0).to(gray_tensor.device)
        return (out_tensor,)

class GenerateColorPalette:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hex_colors": ("STRING", {"multiline": True, "tooltip": "十六进制颜色代码列表，每行一个，如#06d3f5"}),
                "width": ("INT", {"default": 128, "min": 1, "max": 2048, "step": 1, "tooltip": "色卡宽度"}),
                "height": ("INT", {"default": 384, "min": 1, "max": 2048, "step": 1, "tooltip": "色卡总高度"}),
                "direction": (["vertical", "horizontal"], {"default": "vertical", "tooltip": "色卡排列方向：纵向或横向。"}),
                "debug_colors": ("BOOLEAN", {"default": False, "tooltip": "是否在控制台打印颜色调试信息"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_palette"
    CATEGORY = "comfyui_was_image"
    DESCRIPTION = "根据十六进制颜色代码生成纵向排列的色卡图像。"

    def generate_palette(self, hex_colors, width, height, direction, debug_colors):
        # 解析颜色列表
        if isinstance(hex_colors, str):
            color_list = [x.strip() for x in hex_colors.splitlines() if x.strip()]
        else:
            color_list = hex_colors
        
        if not color_list:
            # 如果没有颜色，返回黑色图像
            img_np = np.zeros((height, width, 3), dtype=np.uint8)
            img_tensor = torch.from_numpy(img_np.astype(np.float32) / 255.0).unsqueeze(0)
            return (img_tensor,)
        
        # 计算每个颜色块的高度
        num_colors = len(color_list)
        block_height = height // num_colors
        
        # 创建图像数组
        img_np = np.zeros((height, width, 3), dtype=np.uint8)
        
        # 填充每个颜色块
        for i, hex_color in enumerate(color_list):
            try:
                rgb = hex_to_rgb(hex_color)
                if debug_colors:
                    print(f"输入颜色 {hex_color} -> RGB: {rgb}")

                if direction == "vertical":
                    # 纵向排列逻辑 (原有逻辑)
                    block_height = height // num_colors
                    start_y = i * block_height
                    end_y = start_y + block_height
                    if i == num_colors - 1: # 最后一个块填满剩余高度
                        end_y = height
                    img_np[start_y:end_y, :, :] = rgb
                elif direction == "horizontal":
                    # 横向排列逻辑 (新增逻辑)
                    block_width = width // num_colors
                    start_x = i * block_width
                    end_x = start_x + block_width
                    if i == num_colors - 1: # 最后一个块填满剩余宽度
                        end_x = width
                    img_np[:, start_x:end_x, :] = rgb
            except ValueError as e:
                # 如果颜色格式错误，使用黑色
                print(f"颜色格式错误: {hex_color}, 使用黑色替代")
                if direction == "vertical":
                    start_y = i * (height // num_colors)
                    end_y = start_y + (height // num_colors)
                    if i == num_colors - 1:
                        end_y = height
                    img_np[start_y:end_y, :, :] = (0, 0, 0)
                elif direction == "horizontal":
                    start_x = i * (width // num_colors)
                    end_x = start_x + (width // num_colors)
                    if i == num_colors - 1:
                        end_x = width
                    img_np[:, start_x:end_x, :] = (0, 0, 0)
        
        # 转换为ComfyUI格式的tensor，确保精度
        # 使用更高精度的转换，避免精度损失
        img_tensor = torch.from_numpy(img_np).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)
        return (img_tensor,)

class CheckPersonInText:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "tooltip": "输入要检测的文本"}),
            }
        }
    
    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("has_person",)
    FUNCTION = "check_person"
    CATEGORY = "comfyui_was_image"
    DESCRIPTION = "检测文本中是否包含人相关的名词，如果包含则返回True，否则返回False。"

    def check_person(self, text):
        # 人相关的名词列表（不区分大小写）
        person_keywords = [
            # 基础人物词汇
            "people", "person", "persons",
            "character", "characters",
            "man", "men",
            "woman", "women",
            "girl", "girls",
            "boy", "boys",
            "couple", "couples",
            "child", "children",
            "kid", "kids",
            "baby", "babies",
            "human", "humans",
            "individual", "individuals",
            "adult", "adults",
            "teenager", "teenagers",
            "elder", "elders",
            "senior", "seniors",
            # 职业/角色 - 执法和安全
            "police", "policeman", "policemen", "policewoman", "policewomen", "cop", "cops",
            "detective", "detectives",
            "soldier", "soldiers", "warrior", "warriors",
            "guard", "guards", "guardian", "guardians",
            "security", "bodyguard", "bodyguards",
            "firefighter", "firefighters", "fireman", "firemen",
            "sheriff", "sheriffs", "marshal", "marshals",
            # 职业/角色 - 医疗
            "doctor", "doctors", "physician", "physicians",
            "nurse", "nurses",
            "surgeon", "surgeons",
            "paramedic", "paramedics",
            # 职业/角色 - 教育和学术
            "teacher", "teachers", "professor", "professors",
            "student", "students", "pupil", "pupils",
            "scientist", "scientists", "researcher", "researchers",
            "scholar", "scholars",
            # 职业/角色 - 交通和运输
            "pilot", "pilots",
            "astronaut", "astronauts", "cosmonaut", "cosmonauts",
            "driver", "drivers",
            "captain", "captains", "sailor", "sailors",
            # 职业/角色 - 餐饮和服务
            "chef", "chefs", "cook", "cooks",
            "waiter", "waiters", "waitress", "waitresses",
            "server", "servers", "bartender", "bartenders",
            # 职业/角色 - 艺术和娱乐
            "artist", "artists", "painter", "painters",
            "musician", "musicians", "singer", "singers",
            "actor", "actors", "actress", "actresses",
            "dancer", "dancers",
            "performer", "performers",
            "director", "directors",
            # 职业/角色 - 商业和职业
            "businessman", "businessmen", "businesswoman", "businesswomen",
            "worker", "workers", "employee", "employees",
            "engineer", "engineers",
            "lawyer", "lawyers", "attorney", "attorneys",
            "judge", "judges",
            "farmer", "farmers",
            "builder", "builders", "construction", "worker",
            # 职业/角色 - 领导和政治
            "president", "presidents",
            "leader", "leaders",
            "king", "kings", "queen", "queens",
            "prince", "princes", "princess", "princesses",
            "emperor", "emperors", "empress", "empresses",
            "lord", "lords", "lady", "ladies",
            "duke", "dukes", "duchess", "duchesses",
            # 职业/角色 - 历史和奇幻
            "knight", "knights",
            "wizard", "wizards", "witch", "witches", "sorcerer", "sorcerers",
            "priest", "priests", "monk", "monks", "nun", "nuns",
            "elf", "elves", "dwarf", "dwarves", "dwarfs",
            "hero", "heroes", "heroine", "heroines",
            "superhero", "superheroes", "superheroine", "superheroines",
            "villain", "villains",
            # 职业/角色 - 其他
            "thief", "thieves", "robber", "robbers", "burglar", "burglars",
            "assassin", "assassins",
            "ninja", "ninjas",
            "samurai", "samurais",
            "viking", "vikings",
            "pirate", "pirates",
            "cowboy", "cowboys", "cowgirl", "cowgirls",
            "clown", "clowns",
            "mime", "mimes",
            # 特殊角色
            "android", "androids",
            "doll", "dolls",  # 人偶
            "puppet", "puppets",  # 木偶
        ]
        
        if not text or not isinstance(text, str):
            return (False,)
        
        # 将文本转换为小写以便不区分大小写匹配
        text_lower = text.lower()
        
        # 使用单词边界匹配，确保准确匹配完整单词
        for keyword in person_keywords:
            # 使用单词边界 \b 来匹配完整单词
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, text_lower):
                return (True,)
        
        return (False,)

NODE_CLASS_MAPPINGS = {
    "Replace Color By Palette": ReplaceColorByPalette,
    "ConvertGrayToImage": ConvertGrayToImage,
    "Generate Color Palette": GenerateColorPalette,
    "Check Person In Text": CheckPersonInText,
}
