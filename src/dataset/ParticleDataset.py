from torch.utils.data import Dataset
import torch
import h5py
import os
import numpy as np
from collections import namedtuple
from functools import lru_cache

from src.util.create_random_masks import get_subjets

JETCLASS_QCD_IDX = 0
JETCLASS_TBQQ_IDX = 8

DataSample = namedtuple("DataSample", ["p4_spatial", "p4", "mask"])
DataSample_label = namedtuple("DataSample_label", ["p4_spatial", "p4", "mask", "labels"])

class ParticleDataset(Dataset):
    """
    description:
        ParticleDataset contains the following optimizations:
            - Content cache:  preloads small files into CPU RAM up to cache_size_gb
            - LRU file cache: keeps up to 8 HDF5 files open to avoid reopening
            - For uncached files: reads only the single requested jet 
    """
    def __init__(
        self,
        directory_path,
        num_jets=None,
        return_labels=False,
        label_mode="auto", # which configuration to use
        cache_size_gb=0.0,
        size_multiplier=1.0,
        compute_subjets=False,
        base_seed=42,
        shuffle_files_each_epoch=True
    ):
        self.return_labels = return_labels
        self.size_multiplier = size_multiplier
        self.compute_subjets = compute_subjets
        self.subjets_cache = {}
        self.base_seed = int(base_seed)
        self.shuffle_files_each_epoch = bool(shuffle_files_each_epoch)
        self.epoch = 0
        self.files = sorted(
            os.path.join(directory_path, f)
            for f in os.listdir(directory_path)
            if f.endswith((".h5", ".hdf5"))
        )
        if not self.files:
            raise ValueError(f"No HDF5 files in {directory_path!r}")
        with h5py.File(self.files[0], 'r') as f0:
            stats = {k: f0['stats'][k][:] for k in f0['stats']}

            if label_mode == "auto":
                if "JetClass" in str(directory_path): # pretraining on only Top + QCD jets
                    self.label_mode = "jetclass_top_vs_qcd"
                else:
                    self.label_mode = "synthetic_bkg_vs_sig"
            else:
                self.label_mode = label_mode

        self.mean_log_e, self.std_log_e = stats['part_e_log']
        self.stats = stats
        lengths = []
        self.valid_indices_per_file = []

        for fn in self.files:
            with h5py.File(fn, 'r') as f:
                if self.label_mode == "jetclass_top_vs_qcd":
                    labels = f["labels"][:]          # shape (N, 10)
                    tb  = labels[:, JETCLASS_TBQQ_IDX]
                    qcd = labels[:, JETCLASS_QCD_IDX]

                    valid = (tb == 1) | (qcd == 1)
                    idxs = np.nonzero(valid)[0].astype(np.int64)
                    self.valid_indices_per_file.append(idxs)
                    lengths.append(len(idxs))
                else:
                    L = int(f['labels'].shape[0])
                    self.valid_indices_per_file.append(None)
                    lengths.append(L)

        if num_jets is not None and num_jets < sum(lengths):
            capped_lengths = []
            capped_valid_indices = []
            total = 0

            for fn, L, idxs in zip(self.files, lengths, self.valid_indices_per_file):
                if total >= num_jets:
                    break
                take = min(L, num_jets - total)
                capped_lengths.append(take)
                if idxs is not None:
                    capped_valid_indices.append(idxs[:take])
                else:
                    capped_valid_indices.append(None)
                total += take

            lengths = capped_lengths
            self.files = self.files[:len(lengths)]
            self.valid_indices_per_file = capped_valid_indices
        if self.label_mode == "jetclass_top_vs_qcd":
            total = int(np.sum(lengths))
            per_file_nonzero = sum(int(l > 0) for l in lengths)
            print(f"[ParticleDataset] jetclass_top_vs_qcd enabled")
            print(f"[ParticleDataset] total kept jets = {total}")
            print(f"[ParticleDataset] files with >=1 kept jet = {per_file_nonzero}/{len(lengths)}")
        self.file_lengths = np.array(lengths, dtype=int)
        self.cum_lengths = np.concatenate([[0], np.cumsum(self.file_lengths)])
        self._total = int(self.cum_lengths[-1])
        self.file_to_index = {fn: i for i, fn in enumerate(self.files)}

        self._base_files = list(self.files)
        self._base_valid_indices_per_file = list(self.valid_indices_per_file)
        self._base_file_lengths = self.file_lengths.copy()

        self.cache_size_bytes = int(cache_size_gb * 1024**3)
        self.content_cache = {}
        self.total_cached = 0
        self._cache_order = []
        self.set_epoch(0)
        self._preload_content()

    def _rebuild_indexing(self):
        self.file_lengths = np.array(self.file_lengths, dtype=int)
        self.cum_lengths = np.concatenate([[0], np.cumsum(self.file_lengths)])
        self._total = int(self.cum_lengths[-1])
        self.file_to_index = {fn: i for i, fn in enumerate(self.files)}

    def _clear_epoch_dependent_caches(self):
        self.subjets_cache = {}

    def close(self):
        try:
            self._get_file_handle.cache_clear()
        except Exception:
            pass

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)
        if not self.shuffle_files_each_epoch:
            return

        rng = np.random.RandomState(self.base_seed + self.epoch)
        perm = rng.permutation(len(self._base_files))

        # permute all per-file aligned structures together
        self.files = [self._base_files[i] for i in perm]
        self.valid_indices_per_file = [self._base_valid_indices_per_file[i] for i in perm]
        self.file_lengths = self._base_file_lengths[perm]

        self._rebuild_indexing()
        try:
            self._get_file_handle.cache_clear()
        except Exception:
            pass
            
    def _estimate_size(self, path: str) -> int:
        with h5py.File(path, 'r') as f:
            if self.label_mode == "jetclass_top_vs_qcd":
                n_jets = f['labels'].shape[0]
                n_parts = f['mask'].shape[1]
                bytes_labels = (n_jets * f['labels'].shape[1] * f['labels'].dtype.itemsize) if self.return_labels else 0

            else:
                n_jets = f['labels'].shape[0]
                n_parts = f['mask'].shape[1]
                bytes_labels = (n_jets * f['labels'].shape[1] * f['labels'].dtype.itemsize) if self.return_labels else 0

            bytes_p4_spatial = n_jets * n_parts * 4 * 4
            bytes_p4         = n_jets * n_parts * 4 * 4
            bytes_mask       = n_jets * n_parts * 1 * 4

        return int((bytes_p4_spatial + bytes_p4 + bytes_mask + bytes_labels) * self.size_multiplier)

    def _preload_content(self):
        if self.cache_size_bytes <= 0:
            return
        for fn in sorted(self.files, key=self._estimate_size):
            est = self._estimate_size(fn)
            if self.total_cached + est > self.cache_size_bytes:
                break
            with h5py.File(fn, 'r') as f:
                idxs = None
                if self.label_mode == "jetclass_top_vs_qcd":
                    file_idx = self.file_to_index[fn]
                    idxs = self.valid_indices_per_file[file_idx]

                if idxs is None:
                    parts = {k: f['particles'][k][:] for k in f['particles']}
                    mask_np = f['mask'][:]
                else:
                    parts = {k: f['particles'][k][idxs] for k in f['particles']}
                    mask_np = f['mask'][idxs]

                labels_np = None
                if self.return_labels:
                    if self.label_mode == "jetclass_top_vs_qcd":
                        labels_all = f['labels'][idxs] if idxs is not None else f['labels'][:]
                        tb  = labels_all[:, JETCLASS_TBQQ_IDX]
                        qcd = labels_all[:, JETCLASS_QCD_IDX]
                        labels_np = np.where(tb == 1, 1, 0).astype(np.int64)
                    else:
                        labels_np = f['labels'][:]
                        
            for k, arr in parts.items():
                parts[k] = arr.astype(np.float32)
            mask_np = mask_np.astype(np.float32)
            log_e = parts['part_e_log'] * self.std_log_e + self.mean_log_e
            norm_e = (np.exp(log_e) * mask_np).astype(np.float32)
            p4_spatial_np = np.stack([parts['part_px'], parts['part_py'], parts['part_pz'], norm_e], axis=-1)
            p4_np         = np.stack([parts['part_deta'], parts['part_dphi'], parts['part_pt_log'], parts['part_e_log']], axis=-1)
            p4_spatial = torch.from_numpy(p4_spatial_np)
            p4         = torch.from_numpy(p4_np)
            mask       = torch.from_numpy(mask_np).unsqueeze(-1)
            labels = torch.from_numpy(labels_np).long() if (self.return_labels and labels_np is not None) else None
            data = {'p4_spatial': p4_spatial, 'p4': p4, 'mask': mask, 'labels': labels}
            actual = sum(t.element_size() * t.numel() for t in data.values() if t is not None)
            self.content_cache[fn] = data
            self.total_cached += actual
            self._cache_order.append(fn)

    @lru_cache(maxsize=8)
    def _get_file_handle(self, fn: str) -> h5py.File:
        return h5py.File(fn, 'r', rdcc_nbytes=512*1024**2, rdcc_nslots=1_000_000, rdcc_w0=0.9)

    def _prefetch_file(self, fn: str):
        if self.cache_size_bytes <= 0:
            return
        if fn in self.content_cache:
            return
        with h5py.File(fn, 'r') as f:
            idxs = None
            if self.label_mode == "jetclass_top_vs_qcd":
                file_idx = self.file_to_index[fn]
                idxs = self.valid_indices_per_file[file_idx]

            if idxs is None:
                parts = {k: f['particles'][k][:] for k in f['particles']}
                mask_np = f['mask'][:]
            else:
                parts = {k: f['particles'][k][idxs] for k in f['particles']}
                mask_np = f['mask'][idxs]

            labels_np = None
            if self.return_labels:
                if self.label_mode == "jetclass_top_vs_qcd":
                    labels_all = f['labels'][idxs] if idxs is not None else f['labels'][:]
                    tb  = labels_all[:, JETCLASS_TBQQ_IDX]
                    qcd = labels_all[:, JETCLASS_QCD_IDX]
                    labels_np = np.where(tb == 1, 1, 0).astype(np.int64)
                else:
                    labels_np = f['labels'][:]
                
        for k, arr in parts.items():
            parts[k] = arr.astype(np.float32)
        mask_np = mask_np.astype(np.float32)
        log_e = parts['part_e_log'] * self.std_log_e + self.mean_log_e
        norm_e = (np.exp(log_e) * mask_np).astype(np.float32)
        p4_spatial_np = np.stack([parts['part_px'], parts['part_py'], parts['part_pz'], norm_e], axis=-1)
        p4_np         = np.stack([parts['part_deta'], parts['part_dphi'], parts['part_pt_log'], parts['part_e_log']], axis=-1)
        p4_spatial = torch.from_numpy(p4_spatial_np)
        p4         = torch.from_numpy(p4_np)
        mask       = torch.from_numpy(mask_np).unsqueeze(-1)
        labels = torch.from_numpy(labels_np).long() if (self.return_labels and labels_np is not None) else None
        data = {'p4_spatial': p4_spatial, 'p4': p4, 'mask': mask, 'labels': labels}
        need = sum(t.element_size() * t.numel() for t in data.values() if t is not None)
        while self._cache_order and self.total_cached + need > self.cache_size_bytes:
            victim = self._cache_order.pop(0)
            ev = self.content_cache.pop(victim, None)
            if ev is not None:
                self.total_cached -= sum(t.element_size() * t.numel() for t in ev.values() if t is not None)
        self.content_cache[fn] = data
        self.total_cached += need
        self._cache_order.append(fn)

    def __len__(self):
        return self._total

    def __getitem__(self, idx):
        file_idx = int(np.searchsorted(self.cum_lengths, idx, side='right') - 1)
        local_idx = int(idx - self.cum_lengths[file_idx])
        fn = self.files[file_idx]

        true_idx = local_idx
        if self.label_mode == "jetclass_top_vs_qcd":
            true_idx = int(self.valid_indices_per_file[file_idx][local_idx])
        if fn in self.content_cache:
            d = self.content_cache[fn]
            p_spatial = d['p4_spatial'][local_idx]
            p4_tensor = d['p4'][local_idx]
            p_mask    = d['mask'][local_idx]
            labels    = d['labels'][local_idx] if self.return_labels else None
        else:
            if self.cache_size_bytes > 0:
                self._prefetch_file(fn)
            if fn in self.content_cache:
                d = self.content_cache[fn]
                p_spatial = d['p4_spatial'][local_idx]
                p4_tensor = d['p4'][local_idx]
                p_mask    = d['mask'][local_idx]
                labels    = d['labels'][local_idx] if self.return_labels else None
            else:
                f = self._get_file_handle(fn)
                px   = f['particles']['part_px'][true_idx].astype(np.float32)
                py   = f['particles']['part_py'][true_idx].astype(np.float32)
                pz   = f['particles']['part_pz'][true_idx].astype(np.float32)
                deta = f['particles']['part_deta'][true_idx].astype(np.float32)
                dphi = f['particles']['part_dphi'][true_idx].astype(np.float32)
                ptl  = f['particles']['part_pt_log'][true_idx].astype(np.float32)
                elog = f['particles']['part_e_log'][true_idx].astype(np.float32)
                mask_np = f['mask'][true_idx].astype(np.float32)

                if self.return_labels:
                    if self.label_mode == "jetclass_top_vs_qcd":
                        vec = f['labels'][true_idx]           
                        tb  = vec[JETCLASS_TBQQ_IDX]
                        qcd = vec[JETCLASS_QCD_IDX]
                        labels = torch.tensor(1 if tb == 1 else 0, dtype=torch.long)
                    else:
                        labels = torch.from_numpy(f['labels'][true_idx]).long()
                else:
                    labels = None
                log_e = elog * self.std_log_e + self.mean_log_e
                norm_e = (np.exp(log_e) * mask_np).astype(np.float32)
                p_spatial = torch.from_numpy(np.stack([px, py, pz, norm_e], axis=-1))
                p4_tensor = torch.from_numpy(np.stack([deta, dphi, ptl, elog], axis=-1))
                p_mask    = torch.from_numpy(mask_np).unsqueeze(-1)
        p_spatial = p_spatial * p_mask
        p4_tensor = p4_tensor * p_mask
        subjets_info_sorted = None
        if self.compute_subjets:
            cache_key = (fn, true_idx)   
            if cache_key not in self.subjets_cache:
                valid = p_mask.squeeze(-1).bool().numpy()
                arr   = p_spatial[valid].numpy()
                if arr.size == 0:
                    self.subjets_cache[cache_key] = None
                else:
                    px, py, pz, e = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
                    self.subjets_cache[cache_key] = get_subjets(px, py, pz, e, JET_ALGO="CA", jet_radius=0.2)
            subjets_info_sorted = self.subjets_cache[cache_key]
        if self.return_labels:
            return p_spatial, p4_tensor, p_mask, subjets_info_sorted, labels
        return p_spatial, p4_tensor, p_mask, subjets_info_sorted
