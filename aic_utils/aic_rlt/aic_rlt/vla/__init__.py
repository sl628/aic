"""VLA backend registry for RLT.

Usage:
    from aic_rlt.vla import create_vla_backend

    vla = create_vla_backend(
        "xvla",
        device=device,
        model_dir="/home/yifeng/models/xvla-base",
        instruction="Insert SFP cable into NIC port",
        chunk_length=10,
    )

    # or:
    vla = create_vla_backend(
        "pi05",
        device=device,
        checkpoint_dir="/home/yifeng/workspace/pi05_base/pi05_base",
        chunk_length=10,
    )

Supported backends
------------------
"xvla"   XVLABackend  — lerobot/xvla-base (Florence-2, PyTorch)
"pi05"   Pi05Backend  — openpi pi0.5 (PaliGemma, JAX)
"""

from .base import VLABackend

__all__ = [
    "VLABackend",
    "XVLABackend",
    "XVLAWrapper",
    "Pi05Backend",
    "create_vla_backend",
]


def create_vla_backend(backend: str, device, **kwargs) -> VLABackend:
    """Factory function for VLA backends.

    Args:
        backend: "xvla" or "pi05"
        device:  torch.device
        **kwargs: forwarded to the backend constructor

    Returns:
        Initialised VLABackend subclass with embed_dim and num_tokens set.
    """
    if backend == "xvla":
        from .xvla_backend import XVLABackend
        return XVLABackend(device=device, **kwargs)
    elif backend == "pi05":
        from .pi05_backend import Pi05Backend
        return Pi05Backend(device=device, **kwargs)
    else:
        raise ValueError(
            f"Unknown VLA backend '{backend}'. Supported: 'xvla', 'pi05'."
        )


def __getattr__(name):
    """Lazy imports — avoids pulling in lerobot/JAX at module load time."""
    if name == "Pi05Backend":
        from .pi05_backend import Pi05Backend
        return Pi05Backend
    if name == "XVLABackend":
        from .xvla_backend import XVLABackend
        return XVLABackend
    if name == "XVLAWrapper":
        from .xvla_wrapper import XVLAWrapper
        return XVLAWrapper
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
