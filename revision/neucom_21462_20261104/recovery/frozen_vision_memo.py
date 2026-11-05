"""Reuse the frozen visual encoder result for exactly equal model inputs."""

from functools import wraps


class FrozenVisionMemo:
    def __init__(self, adapter, model_id):
        import torch

        self.torch = torch
        self.entries = {}
        self.hits = 0
        self.encodes = 0
        if model_id == "qwen3_vl_4b":
            owner = adapter.model.model
            methods = ("get_image_features",)
        elif model_id == "internvl35_8b":
            owner = adapter.model
            methods = ("extract_feature",)
        elif model_id == "gemma3_12b":
            owner = adapter.model
            methods = ("get_image_features",)
        else:
            raise ValueError(f"Unknown frozen visual encoder {model_id}")
        for name in methods:
            setattr(owner, name, self.memoize(name, getattr(owner, name)))

    def clear(self):
        self.entries.clear()

    def equal(self, current, previous):
        if current is None or previous is None:
            return current is previous
        return (
            self.torch.is_tensor(current)
            and self.torch.is_tensor(previous)
            and current.dtype == previous.dtype
            and current.device == previous.device
            and current.shape == previous.shape
            and self.torch.equal(current, previous)
        )

    def memoize(self, name, encode):
        @wraps(encode)
        def wrapped(*args, **kwargs):
            if name in self.entries:
                old_args, old_kwargs, result = self.entries[name]
                if (
                    len(args) == len(old_args)
                    and kwargs.keys() == old_kwargs.keys()
                    and all(self.equal(a, b) for a, b in zip(args, old_args))
                    and all(self.equal(kwargs[k], old_kwargs[k]) for k in kwargs)
                ):
                    self.hits += 1
                    return result
            result = encode(*args, **kwargs)
            self.encodes += 1
            self.entries[name] = (args, kwargs, result)
            return result

        return wrapped
