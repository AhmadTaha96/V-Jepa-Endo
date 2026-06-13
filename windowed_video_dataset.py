# ============================================================================
# windowed_video_dataset.py
# ----------------------------------------------------------------------------
# Per-sample randomized temporal window for V-JEPA 2.1 endoscopic pretraining.
#
# The repo's VideoDataset fixes the temporal stride per dataset: with fps mode
# the window is always fpc/fps seconds (native_fps cancels), so you cannot vary
# how much real time a 16-frame clip covers. This subclass instead samples a
# target WINDOW in SECONDS per __getitem__ and converts it to a per-file stride
# using that file's native fps:
#
#     stride = round(window_seconds * native_fps / fpc)
#
# so "8 seconds" means 8 seconds of procedure time regardless of whether the
# source is 25 or 30 fps. A batch then mixes wide (sparse, long-context) and
# narrow (dense, short-context) clips. Everything downstream (collator, masks,
# model) is unchanged; it still receives a (B, 3, fpc, H, W) tensor.
#
# Only loadvideo_decord's stride computation differs from the parent. The rest
# of the body is reproduced verbatim so clip-window sampling stays identical.
# ============================================================================

import math
import warnings

import numpy as np
import torch
from decord import cpu, VideoReader

from src.datasets.video_dataset import VideoDataset, make_videodataset
from src.datasets.utils.weighted_sampler import DistributedWeightedSampler
from src.datasets.utils.dataloader import NondeterministicDataLoader


class WindowedVideoDataset(VideoDataset):
    """VideoDataset that picks a temporal window (in seconds) per sample and
    derives the frame stride from each file's own native fps.

    window_seconds:
        - a single float  -> fixed window, but expressed in real time
        - a list/tuple     -> one value drawn uniformly at random per sample
    min_stride: floor so dense windows never collapse to stride 0 on low-fps files.
    """

    def __init__(self, *args, window_seconds=(1.0, 4.0, 8.0, 12.0, 16.0), min_stride=1, **kwargs):
        # parent requires exactly one of {fps, duration, frame_step}; we pass a
        # placeholder frame_step and then take over stride ourselves
        kwargs.setdefault("frame_step", 1)
        kwargs["fps"] = None
        kwargs["duration"] = None
        super().__init__(*args, **kwargs)
        if isinstance(window_seconds, (int, float)):
            window_seconds = [float(window_seconds)]
        self.window_seconds = list(window_seconds)
        self.min_stride = int(min_stride)

    def _sample_window_seconds(self):
        return float(np.random.choice(self.window_seconds))

    def loadvideo_decord(self, sample, fpc):
        fname = sample
        if not __import__("os").path.exists(fname):
            warnings.warn(f"video path not found {fname=}")
            return [], None

        _fsize = __import__("os").path.getsize(fname)
        if _fsize > self.filter_long_videos:
            warnings.warn(f"skipping long video of size {_fsize=} (bytes)")
            return [], None

        try:
            vr = VideoReader(fname, num_threads=-1, ctx=cpu(0))
        except Exception:
            return [], None

        # --- the only change vs parent: stride from a real-time window ---
        try:
            video_fps = math.ceil(vr.get_avg_fps())
        except Exception:
            video_fps = 30
        win_s = self._sample_window_seconds()
        fstp = int(round(win_s * video_fps / fpc))
        fstp = max(self.min_stride, fstp)
        # -----------------------------------------------------------------

        clip_len = int(fpc * fstp)

        if self.filter_short_videos and len(vr) < clip_len:
            warnings.warn(f"skipping video of length {len(vr)}")
            return [], None

        vr.seek(0)
        partition_len = len(vr) // self.num_clips

        all_indices, clip_indices = [], []
        for i in range(self.num_clips):
            if partition_len > clip_len:
                end_indx = clip_len
                if self.random_clip_sampling:
                    end_indx = np.random.randint(clip_len, partition_len)
                start_indx = end_indx - clip_len
                indices = np.linspace(start_indx, end_indx, num=fpc)
                indices = np.clip(indices, start_indx, end_indx - 1).astype(np.int64)
                indices = indices + i * partition_len
            else:
                if not self.allow_clip_overlap:
                    indices = np.linspace(0, partition_len, num=partition_len // fstp)
                    indices = np.concatenate(
                        (indices, np.ones(fpc - partition_len // fstp) * partition_len)
                    )
                    indices = np.clip(indices, 0, partition_len - 1).astype(np.int64)
                    indices = indices + i * partition_len
                else:
                    sample_len = min(clip_len, len(vr)) - 1
                    indices = np.linspace(0, sample_len, num=sample_len // fstp)
                    indices = np.concatenate(
                        (indices, np.ones(fpc - sample_len // fstp) * sample_len)
                    )
                    indices = np.clip(indices, 0, sample_len - 1).astype(np.int64)
                    clip_step = 0
                    if len(vr) > clip_len:
                        clip_step = (len(vr) - clip_len) // (self.num_clips - 1)
                    indices = indices + i * clip_step

            clip_indices.append(indices)
            all_indices.extend(list(indices))

        buffer = vr.get_batch(all_indices).asnumpy()
        return buffer, clip_indices


def make_windowed_videodataset(
    data_paths,
    batch_size,
    window_seconds,
    dataset_fpcs,
    transform=None,
    shared_transform=None,
    rank=0,
    world_size=1,
    datasets_weights=None,
    collator=None,
    drop_last=True,
    num_workers=10,
    pin_mem=True,
    persistent_workers=True,
    deterministic=True,
    min_stride=1,
    filter_short_videos=False,
    filter_long_videos=int(10**9),
    **_ignored,
):
    """Mirror of make_videodataset but instantiates WindowedVideoDataset.
    Returns (dataset, data_loader, dist_sampler) like the original."""
    dataset = WindowedVideoDataset(
        data_paths=data_paths,
        datasets_weights=datasets_weights,
        dataset_fpcs=dataset_fpcs,
        window_seconds=window_seconds,
        min_stride=min_stride,
        shared_transform=shared_transform,
        transform=transform,
        filter_short_videos=filter_short_videos,
        filter_long_videos=filter_long_videos,
    )

    if datasets_weights is not None:
        dist_sampler = DistributedWeightedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
    else:
        dist_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True
        )

    loader_cls = (
        torch.utils.data.DataLoader if deterministic else NondeterministicDataLoader
    )
    data_loader = loader_cls(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0) and persistent_workers,
    )
    return dataset, data_loader, dist_sampler
