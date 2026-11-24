"""Exploration maps drawn while training. Every `exploration_map_interval` rounds the trainer copies what the map
needs from its in-memory replay buffer (diagnostics/images/exploration_map.py: snapshot_replay_buffer, a few columns
of every row and a couple of frames per 2 x 2 block of player positions) and draw_snapshot() draws it in a worker
process at below-normal priority, while training goes on. The image is
diagnostics/images/exploration_map_G<update>.png (the same map exploration_image.ipynb draws by hand from a saved
buffer), and the newest one is also copied to plots/exploration_map.png."""
import multiprocessing as mp
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent                       # project root (holds gq_dreamer/)
MAP_DIR = ROOT / "diagnostics" / "images"                           # exploration_map.py and the map history


def _exploration_module():
    """diagnostics/images/exploration_map.py as a module."""
    if str(MAP_DIR) not in sys.path:
        sys.path.insert(0, str(MAP_DIR))
    import exploration_map
    return exploration_map


def _below_normal_priority():
    """Lowers this process's priority so play and training keep theirs."""
    try:
        if sys.platform == "win32":
            import ctypes
            k32 = ctypes.windll.kernel32
            k32.SetPriorityClass(k32.GetCurrentProcess(), 0x4000)  # BELOW_NORMAL_PRIORITY_CLASS
        else:
            os.nice(5)
    except Exception:                                                # priority is a nicety: never fail on it
        pass


def _draw(snap, update, out, latest):
    """Worker process: draws the snapshot to `out` and copies it to `latest` -> image path."""
    _below_normal_priority()
    path = _exploration_module().draw_snapshot(snap, out, source="replay buffer", update=update)
    if latest:
        os.makedirs(os.path.dirname(os.path.abspath(latest)), exist_ok=True)
        tmp = os.path.join(os.path.dirname(os.path.abspath(latest)), f".{os.path.basename(latest)}.tmp.png")
        shutil.copyfile(path, tmp)
        os.replace(tmp, latest)                                      # a viewer never sees a half-written PNG
    return path


class ExplorationMapJob:
    """One exploration map at a time, drawn by a worker process (spawned on first use, kept for the next maps)."""

    def __init__(self, latest="plots/exploration_map.png", out_dir=MAP_DIR):
        self.latest = latest                                         # copy of the newest map ("" = none)
        self.out_dir = Path(out_dir)                                 # exploration_map_G<update>.png history
        self._pool = None
        self._future = None
        self.update = None                                           # gradient update of the map being drawn
        self.summary = ""

    @property
    def busy(self):
        return self._future is not None and not self._future.done()

    def submit(self, buffer, update):
        """Copies what the map needs from the buffer and starts drawing it -> False when the previous map is still
        being drawn (this one is skipped)."""
        if self.busy:
            return False
        em = _exploration_module()
        snap = em.snapshot_replay_buffer(buffer)
        if not len(snap["ep"]):
            raise ValueError("the replay buffer is empty")
        out = str(self.out_dir / f"exploration_map_G{int(update)}.png")
        if self._pool is None:
            self._pool = ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn"))
        self._future = self._pool.submit(_draw, snap, int(update), out, self.latest)
        self.update = int(update)
        self.summary = (f"{len(np.unique(snap['ep']))} episodes, {len(np.unique(snap['maps']))} maps, "
                        f"{len(snap['frames']):,} frames painted")
        return True

    def poll(self):
        """None while nothing has finished, else (path or None, summary or error text)."""
        if self._future is None or not self._future.done():
            return None
        fut, self._future = self._future, None
        exc = fut.exception()
        if exc is not None:
            if isinstance(exc, BrokenProcessPool):
                self.close()                                         # a dead worker: start a new one next time
            return None, f"{type(exc).__name__}: {exc}"
        return fut.result(), self.summary

    def close(self, wait=False):
        """Stops the worker once the map being drawn is done (wait=False: training does not wait for it)."""
        if self._pool is not None:
            self._pool.shutdown(wait=wait)
            self._pool = None
