"""Dream visualisation: decoded dream frames (72x80) with actions, predicted rewards, new memory bits, curiosity and
advantages. The GPU part (decoding, heads) runs here; the drawing is dream_render's (matplotlib only, so it can run in
a worker process)."""
import os

import numpy as np
import torch

from .dream_render import render_dream, render_dreams_pdf
from .model import PixelDecoder


@torch.no_grad()
def dream_data(agent, dream):
    """One packed dream (agent.imagine) -> numpy data for dream_render: decoded frames, grounding-head map / battle
    flag, curiosity, memory bits held."""
    st = dream["states"].to(agent.device).float()
    _, zf, mem = agent.split_state(st)
    with agent._amp():
        frames = PixelDecoder.to_image(agent.tokenizer.decode(zf)).cpu()
        maps, battle = agent.tokenizer.heads.predict(zf)               # grounding heads read z only
    curiosities = agent.two_hot.decode(agent.curiosity_head(st)).squeeze(-1).cpu()
    to_np = lambda x: torch.as_tensor(x).detach().float().cpu().numpy()   # noqa: E731
    return {"frames": np.clip(frames.float().numpy(), 0, 1), "maps": maps.cpu().numpy(),
            "battle": battle.cpu().numpy(), "n_mem": mem.sum(-1).cpu().numpy(), "curiosity": curiosities.numpy(),
            "rewards": to_np(dream["rewards"]), "events": to_np(dream["events"]),
            "actions": torch.as_tensor(dream["actions"]).argmax(-1).cpu().numpy(),
            "advantages": to_np(dream["advantages"]), "metric": float(dream["metric"]), "label": dream.get("label")}


def plot_dream(agent, dream, pdf=None):
    """Draws one packed dream (agent.imagine) - decoded frames + per-step info; adds it to `pdf` or shows it."""
    render_dream(dream_data(agent, dream), pdf=pdf)


def save_dreams_pdf(agent, dreams, directory, keep=10, executor=None):
    """Writes packed dreams to <directory>/dream_g<updates>.pdf, keeping the newest `keep` PDFs -> path. executor: the
    drawing is submitted to it (a worker process) and the PDF appears a few seconds later; the future is stored in
    save_dreams_pdf.pending."""
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"dream_g{agent.total_num_updates}.pdf")
    data = [dream_data(agent, d) for d in dreams]
    if executor is None:
        return render_dreams_pdf(path, data, directory, keep)
    save_dreams_pdf.pending = executor.submit(render_dreams_pdf, path, data, directory, keep)
    return path


save_dreams_pdf.pending = None
