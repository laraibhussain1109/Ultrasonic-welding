"""Temporary disk storage for training crops and unsampled patch embeddings."""

from pathlib import Path

import numpy as np


class DiskPatchEmbeddings:
    """Feed the unchanged seeded coreset sampler without concatenating RAM copies."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle = path.open("wb")
        self.count = 0
        self.columns = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._handle.close()

    def append(self, rows) -> None:
        values = np.ascontiguousarray(rows.detach().cpu().numpy(), dtype=np.float32)
        if values.ndim != 2 or (self.columns is not None and values.shape[1] != self.columns):
            raise ValueError("Training patch dimensions changed")
        self.columns = values.shape[1]
        values.tofile(self._handle)
        self.count += len(values)

    def build_memory(self, inspector, torch):
        self._handle.flush()
        if not self.count:
            raise ValueError("No qualified training patches remain")
        mapping = np.memmap(self.path, dtype=np.float32, mode="r+", shape=(self.count, self.columns))
        tensor = torch.from_numpy(mapping)
        try:
            # Only the sampler's bounded candidate set and final coreset are
            # allocated in RAM. The full feature pool remains disk-backed.
            return inspector._build_patchcore_memory(tensor, torch)
        finally:
            del tensor
            mapping._mmap.close()
