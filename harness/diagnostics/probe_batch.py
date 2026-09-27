"""Fixed GPU probe batch for tier 2(b) (ANALYSIS_SUITE_SPEC §6b)."""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

import config
import data


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_val_item(item_index: int, *, probe_slot: int = 0):
    root = str(_repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)
    from analysis.load_sample import load_synthetic_sample, load_worldsnap_sample
    if config.data_source == "synthetic":
        return load_synthetic_sample(split="val", item_index=item_index, seed=config.seed)
    if config.task_rung == "c1":
        from analysis.load_c1_sample import load_c1_worldsnap_sample
        task_key = config.c1_task_key
        if config.c1_is_multi_task():
            keys = config.c1_task_keys_active()
            task_key = keys[probe_slot % len(keys)]
        return load_c1_worldsnap_sample(
            split="val",
            crop_index=item_index,
            task_key=task_key,
            encoding=config.encoding_mode,
            img_size=config.img_size,
            grid_size=config.grid_out_size,
        )
    if config.task_rung == "c2":
        from analysis.load_c2_sample import load_c2_worldsnap_sample
        keys = config.c2_task_keys_active()
        return load_c2_worldsnap_sample(
            split="val",
            crop_index=item_index,
            task_key=keys[probe_slot % len(keys)],
            encoding=config.encoding_mode,
            img_size=config.img_size,
            grid_size=config.grid_out_size,
            cropset=config.cropset,
        )
    if config.task_rung == "c3":
        from analysis.load_c3_sample import load_c3_worldsnap_sample
        return load_c3_worldsnap_sample(
            split="val",
            crop_index=item_index,
            task_key=config.c3_eval_key(),
            encoding=config.encoding_mode,
            img_size=config.img_size,
            grid_size=config.grid_out_size,
            cropset=config.cropset,
        )
    if config.task_rung == "c4":
        raise RuntimeError("c4 probe uses ProbeBatch.build() val_loader.fetch_at")
    return load_worldsnap_sample(
        split="val",
        item_index=item_index,
        encoding=config.encoding_mode,
        img_size=config.img_size,
        grid_size=config.grid_out_size,
    )


@dataclass
class ProbeBatch:
    img: torch.Tensor
    tgt: torch.Tensor
    marker_row: torch.Tensor
    marker_col: torch.Tensor
    item_indices: list[int]
    pred_indices: list[int]
    task_ids: torch.Tensor | None = None
    cond_ids: torch.Tensor | None = None

    @classmethod
    def build(cls, device: torch.device | None = None) -> ProbeBatch:
        device = device or data.device
        n_probe = config.diagnostics_probe_batch_size
        n_pred = min(config.diagnostics_pred_samples, n_probe)
        base_seed = config.diagnostics_probe_seed
        rng_indices = list(range(n_probe))
        if config.data_source == "worldsnap":
            max_val = data.val_crop_count()
            rng_indices = [(base_seed * 997 + i * 17) % max_val for i in range(n_probe)]
        pred_indices = list(range(n_pred))
        imgs, tgts, mrs, mcs, tids, cids = [], [], [], [], [], []
        h, w = config.img_size, config.img_size
        multi_c1 = config.task_rung == "c1" and config.c1_is_multi_task()
        multi_c2 = config.task_rung == "c2"
        is_c3 = config.task_rung == "c3"
        is_c4 = config.task_rung == "c4"
        if is_c4:
            return cls.build_c4_from_val_loader(data.val_loader, device=device)
        for slot, idx in enumerate(rng_indices):
            sample = _load_val_item(idx, probe_slot=slot)
            imgs.append(sample.img)
            tgts.append(sample.target)
            if config.task_rung == "c1":
                mrs.append(h // 2)
                mcs.append(w // 2)
            elif config.task_rung == "c2":
                mrs.append(h // 2)
                mcs.append(w // 2)
            elif config.task_rung == "c3":
                from spatial_data.c3_tasks import get_c3_task
                if get_c3_task(config.c3_eval_key()).show_marker_plane:
                    mrs.append(int(sample.meta.marker_row))
                    mcs.append(int(sample.meta.marker_col))
                else:
                    mrs.append(h // 2)
                    mcs.append(w // 2)
            else:
                mrs.append(sample.meta.marker_row)
                mcs.append(sample.meta.marker_col)
            if multi_c1:
                keys = config.c1_task_keys_active()
                tids.append(slot % len(keys))
            if multi_c2:
                keys = config.c2_task_keys_active()
                ti = slot % len(keys)
                tids.append(ti)
                cids.append(sample.cond_ids)
            if is_c3:
                tids.append(0)
                cids.append(sample.cond_ids)
        img = torch.stack(imgs).to(device)
        tgt = torch.stack(tgts).to(device)
        task_ids = torch.tensor(tids, device=device, dtype=torch.long) if tids else None
        cond_ids = torch.tensor(cids, device=device, dtype=torch.long) if cids else None
        return cls(
            img=img,
            tgt=tgt,
            marker_row=torch.tensor(mrs, device=device, dtype=torch.long),
            marker_col=torch.tensor(mcs, device=device, dtype=torch.long),
            item_indices=rng_indices,
            pred_indices=pred_indices,
            task_ids=task_ids,
            cond_ids=cond_ids,
        )

    @classmethod
    def build_c4_from_val_loader(cls, val_loader, *, device: torch.device | None = None) -> ProbeBatch:
        device = device or data.device
        if not hasattr(val_loader, "fetch_at"):
            raise RuntimeError("c4 diagnostics need C4 val loader with fetch_at")
        n_probe = config.diagnostics_probe_batch_size
        n_pred = min(config.diagnostics_pred_samples, n_probe)
        base_seed = config.diagnostics_probe_seed
        n_val = len(val_loader)
        imgs, tgts, mrs, mcs, tids, cids, rng_indices = [], [], [], [], [], [], []
        for slot in range(n_probe):
            flat_ix = (base_seed * 997 + slot * 17) % n_val
            img1, tgt1, tid1, cid1, meta = val_loader.fetch_at(flat_ix)
            imgs.append(img1.cpu())
            tgts.append(tgt1.cpu())
            mk = meta["markers"][0].cpu().numpy()
            mcs.append(int(mk[0, 0]))
            mrs.append(int(mk[0, 1]))
            tids.append(int(tid1.item()))
            cids.append(cid1.cpu().tolist())
            rng_indices.append(flat_ix)
        img = torch.stack(imgs).to(device)
        tgt = torch.stack(tgts).to(device)
        return cls(
            img=img, tgt=tgt,
            marker_row=torch.tensor(mrs, device=device, dtype=torch.long),
            marker_col=torch.tensor(mcs, device=device, dtype=torch.long),
            item_indices=rng_indices,
            pred_indices=list(range(n_pred)),
            task_ids=torch.tensor(tids, device=device, dtype=torch.long),
            cond_ids=torch.tensor(cids, device=device, dtype=torch.long),
        )

    def marker_token_indices(self, cfg) -> torch.Tensor:
        """Attention-matrix column of the marker's patch token.

        With cond_tokens conditioning the encoder sequence starts with n_cond_token_slots
        task tokens, so patch i sits in column i + n_prefix. Without the offset the probe
        read attention to the patch two positions earlier (found 23 Sep: attn_to_marker
        showed ~0.0 on runs whose block-1 heads put 54-62 % of their mass on the marker).
        """
        g = cfg.img_size // cfg.patch_size
        pr = self.marker_row // cfg.patch_size
        pc = self.marker_col // cfg.patch_size
        n_prefix = int(getattr(cfg, "n_cond_token_slots", 0))
        return (pr * g + pc + n_prefix).long()

    def update_meta(self, meta_path: Path) -> None:
        if not meta_path.is_file():
            return
        doc = json.loads(meta_path.read_text(encoding="utf-8"))
        doc["probe_sample_ids"] = self.item_indices
        doc["pred_sample_ids"] = [self.item_indices[i] for i in self.pred_indices]
        meta_path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
