
from clip import CLIP
from encoder import VAE_Encoder
from decoder import VAE_Decoder
from diffusion import Diffusion

import model_converter   # IMPORTANT: load_from_standard_weights is here, not in model_loader

def preload_models_from_standard_weights(ckpt_path, device):
    # Load and convert weights from SD checkpoint
    state_dict = model_converter.load_from_standard_weights(ckpt_path, device)

    # -----------------------
    #        ENCODER
    # -----------------------
    encoder = VAE_Encoder().to(device)

    # Fix keys: prepend "blocks." to match the new VAE_Encoder architecture
    original_encoder_state = state_dict['encoder']
    fixed_encoder_state = {}

    for k, v in original_encoder_state.items():
        fixed_encoder_state[f"blocks.{k}"] = v

    encoder.load_state_dict(fixed_encoder_state, strict=True)

    # -----------------------
    #        DECODER
    # -----------------------
    decoder = VAE_Decoder().to(device)
    decoder.load_state_dict(state_dict['decoder'], strict=True)

    # -----------------------
    #        DIFFUSION (UNet)
    # -----------------------
    diffusion = Diffusion().to(device)
    diffusion.load_state_dict(state_dict['diffusion'], strict=True)

    # -----------------------
    #         CLIP
    # -----------------------
    clip = CLIP().to(device)
    clip.load_state_dict(state_dict['clip'], strict=True)

    return {
        "clip": clip,
        "encoder": encoder,
        "decoder": decoder,
        "diffusion": diffusion,
    }
