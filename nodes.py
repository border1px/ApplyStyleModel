import torch
import os
import sys
import json
import hashlib
import traceback
import math
import time
import random

class StyleModelApply:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"conditioning": ("CONDITIONING", ),
                             "style_model": ("STYLE_MODEL", ),
                             "clip_vision_output": ("CLIP_VISION_OUTPUT", ),
                             "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}),
                             "strength_type": (["multiply", "attn_bias"], ),
                             }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "apply_stylemodel"

    CATEGORY = "conditioning/style_model"

    def apply_stylemodel(self, conditioning, style_model, clip_vision_output, strength, strength_type):
        cond = style_model.get_cond(clip_vision_output).flatten(start_dim=0, end_dim=1).unsqueeze(dim=0)
        if strength_type == "multiply":
            cond *= strength

        n = cond.shape[1]
        c_out = []
        
        print(f"===================== len(conditioning): {len(conditioning)}")
        
        print(f"===================== clip_vision_output: {clip_vision_output}")
        
        # print(f"===================== clip_vision_output.shape: {clip_vision_output.shape}")
        
        print(f"===================== style_model.get_cond(clip_vision_output): {cond}")
        
        for t in conditioning:
            (txt, keys) = t
            keys = keys.copy()
            if strength_type == "attn_bias" and strength != 1.0:
                # math.log raises an error if the argument is zero
                # torch.log returns -inf, which is what we want
                attn_bias = torch.log(torch.Tensor([strength]))
                # get the size of the mask image
                mask_ref_size = keys.get("attention_mask_img_shape", (1, 1))
                                
                n_ref = mask_ref_size[0] * mask_ref_size[1]
                n_txt = txt.shape[1]
                
                print(f"===================== n_txt.shape: {txt.shape}")
                print(f"===================== n_txt.shape[1]: {txt.shape[1]}")
                print(f"===================== conditioning-keys: {keys}")
                
                # grab the existing mask
                mask = keys.get("attention_mask", None)
                
                # create a default mask if it doesn't exist
                if mask is None:
                    mask = torch.zeros((txt.shape[0], n_txt + n_ref, n_txt + n_ref), dtype=torch.float16)
                # convert the mask dtype, because it might be boolean
                # we want it to be interpreted as a bias
                if mask.dtype == torch.bool:
                    # log(True) = log(1) = 0
                    # log(False) = log(0) = -inf
                    mask = torch.log(mask.to(dtype=torch.float16))
                # now we make the mask bigger to add space for our new tokens
                new_mask = torch.zeros((txt.shape[0], n_txt + n + n_ref, n_txt + n + n_ref), dtype=torch.float16)
                # copy over the old mask, in quandrants
                new_mask[:, :n_txt, :n_txt] = mask[:, :n_txt, :n_txt]
                new_mask[:, :n_txt, n_txt+n:] = mask[:, :n_txt, n_txt:]
                new_mask[:, n_txt+n:, :n_txt] = mask[:, n_txt:, :n_txt]
                new_mask[:, n_txt+n:, n_txt+n:] = mask[:, n_txt:, n_txt:]
                # now fill in the attention bias to our redux tokens
                new_mask[:, :n_txt, n_txt:n_txt+n] = attn_bias
                new_mask[:, n_txt+n:, n_txt:n_txt+n] = attn_bias
                keys["attention_mask"] = new_mask.to(txt.device)
                keys["attention_mask_img_shape"] = mask_ref_size

            c_out.append([torch.cat((txt, cond), dim=1), keys])

        return (c_out,)

# Node class mappings
NODE_CLASS_MAPPINGS = {
    "ApplyStyleModel（YPF）": StyleModelApply,
}

# Display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "StyleModelApply": "Apply Style Model",
}