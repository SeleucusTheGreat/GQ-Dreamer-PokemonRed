"""GQ-Dreamer for Pokémon Red: the 72x80 frame -> 25 linear patches + 8 vector tokens (grid, map, status, party,
opponent, items, menu, event memory) -> fusion encoder, 32x32 one-hot latent, conv pixel decoder, grounding heads and
the memory head M on z, TSSM whose input is [z, a, m], agent state s = [h, z, m] with the explicit 144-bit event
memory; dreams pay each memory bit once. Death is not modelled."""
import importlib

# Lazy exports (PEP 562): env worker processes only import gq_dreamer.env, so they do not load torch or matplotlib.
_EXPORTS = {"run": ".trainer", "Trainer": ".trainer", "GQDreamer": ".gq_dreamer", "create_envs": ".env",
            "GQDreamerConfig": ".config", "model_config": ".config", "config_from_dict": ".config",
            "POKEMON_CONFIG": ".config", "LIGHT_CONFIG": ".config", "MODEL_CONFIGS": ".config"}
__all__ = list(_EXPORTS)


def __getattr__(name):
    """Imports an exported name from its module on first use."""
    if name in _EXPORTS:
        value = getattr(importlib.import_module(_EXPORTS[name], __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
