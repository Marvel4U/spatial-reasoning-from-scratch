"""C5 M1: patch_embed weight expansion for extra input channels."""
import torch

from plain_gpt_module.checkpoint import adapt_state_dict_patch_embed_in_chans
from plain_gpt_module.local_config import LocalGridViTConfig
from plain_gpt_module.local_grid_vit import LocalGridViT
from plain_gpt_module.patch_embed import PatchEmbed, expand_proj_weight


def test_expand_proj_weight_16_to_17_zero_scratchpad():
    p, n_embd = 16, 384
    pe16 = PatchEmbed(16, p, n_embd)
    w17 = expand_proj_weight(pe16.proj.weight.data, old_in_chans=16, new_in_chans=17, patch_size=p)
    assert w17.shape == (n_embd, 17 * p * p)
    pp = p * p
    assert torch.allclose(w17[:, : 16 * pp], pe16.proj.weight.data)
    assert w17[:, 16 * pp :].abs().max().item() == 0.0
    pe17 = PatchEmbed(17, p, n_embd)
    pe17.proj.weight.data.copy_(w17)
    pe17.proj.bias.data.copy_(pe16.proj.bias.data)
    img16 = torch.randn(2, 16, 256, 256)
    pad = torch.zeros(2, 1, 256, 256)
    img17 = torch.cat([img16, pad], dim=1)
    out16 = pe16(img16)
    out17 = pe17(img17)
    assert torch.allclose(out16, out17, atol=1e-5, rtol=1e-5)


def test_load_checkpoint_adapts_16_to_17():
    n_embd = 384
    m16 = LocalGridViT(LocalGridViTConfig(in_chans=16, n_embd=n_embd, enc_n_layer=1))
    sd = m16.state_dict()
    m17 = LocalGridViT(LocalGridViTConfig(in_chans=17, n_embd=n_embd, enc_n_layer=1))
    adapted = adapt_state_dict_patch_embed_in_chans(m17, sd, verbose=False)
    m17.load_state_dict(adapted, strict=True)
    img16 = torch.randn(1, 16, 256, 256)
    img17 = torch.cat([img16, torch.zeros(1, 1, 256, 256)], dim=1)
    with torch.no_grad():
        pe16 = m16.patch_embed(img16)
        pe17 = m17.patch_embed(img17)
    assert torch.allclose(pe16, pe17, atol=1e-5, rtol=1e-5)
