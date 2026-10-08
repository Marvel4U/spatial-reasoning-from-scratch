"""C5 M2: readout prefix slot + token CE beside grid loss."""
import torch

from plain_gpt_module.local_config import LocalGridViTConfig
from plain_gpt_module.local_grid_vit import LocalGridViT


def _c4_tok_cfg():
    return LocalGridViTConfig(
        in_chans=17,
        n_tasks=3,
        n_cond_emb=24,
        c1_task_conditioning="cond_tokens",
        token_head_enabled=True,
        n_token_classes=37,
        enc_n_layer=1,
    )


def test_token_head_adds_prefix_slot():
    cfg = _c4_tok_cfg()
    assert cfg.n_prefix_tokens == 3
    assert cfg.encoder_seq_len == 256 + 3
    m = LocalGridViT(cfg)
    B = 2
    img = torch.randn(B, 17, 256, 256)
    tgt = torch.randint(0, 2, (B, 64, 64))
    cond = torch.zeros(B, 2, dtype=torch.long)
    task_id = torch.zeros(B, dtype=torch.long)
    tok = torch.tensor([5, -1], dtype=torch.long)
    logits, loss = m(img, tgt, task_id=task_id, cond_ids=cond, tok_tgt=tok)
    assert logits.shape == (B, 64, 64, 2)
    assert loss.ndim == 0
    loss.backward()


def test_no_token_loss_when_all_ignore():
    cfg = _c4_tok_cfg()
    m = LocalGridViT(cfg)
    B = 2
    img = torch.randn(B, 17, 256, 256)
    tgt = torch.randint(0, 2, (B, 64, 64))
    cond = torch.zeros(B, 2, dtype=torch.long)
    task_id = torch.zeros(B, dtype=torch.long)
    tok = torch.full((B,), -1, dtype=torch.long)
    _, loss = m(img, tgt, task_id=task_id, cond_ids=cond, tok_tgt=tok)
    loss.backward()
