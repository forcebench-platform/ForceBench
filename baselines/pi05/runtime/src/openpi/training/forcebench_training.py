"""Explicit optimization of pretrained weights and newly initialized modules."""
import dataclasses
import jax
import optax

NEW_MODULES = {"force_spec", "force_cross_attn", "ff_head"}


def is_new_path(path):
    return any(str(getattr(item, "key", getattr(item, "name", item))) in NEW_MODULES for item in path)


def parameter_labels(params):
    return jax.tree.map_with_path(lambda path, _: "new" if is_new_path(path) else "backbone", params)


@dataclasses.dataclass(frozen=True)
class ForceBenchAdamW:
    new_module_lr_multiplier: float = 10.0
    b1: float = 0.9
    b2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 1e-10
    clip_gradient_norm: float = 1.0

    def create(self, lr, weight_decay_mask=None):
        if self.new_module_lr_multiplier <= 0:
            raise ValueError("New module LR multiplier must be positive")
        def adam(schedule):
            return optax.adamw(schedule, b1=self.b1, b2=self.b2, eps=self.eps,
                               weight_decay=self.weight_decay, mask=weight_decay_mask)
        return optax.chain(
            optax.clip_by_global_norm(self.clip_gradient_norm),
            optax.multi_transform({"backbone": adam(lr), "new": adam(lambda step: lr(step) * self.new_module_lr_multiplier)}, parameter_labels),
        )


def module_metrics(params, grads, updates):
    result = {}
    for module in sorted(NEW_MODULES):
        if module in params:
            result[f"{module}/grad_norm"] = optax.global_norm(grads[module])
            result[f"{module}/update_norm"] = optax.global_norm(updates[module])
            result[f"{module}/relative_update"] = optax.global_norm(updates[module]) / (optax.global_norm(params[module]) + 1e-12)
    if "force_cross_attn" in params:
        import jax.numpy as jnp
        gate = jnp.tanh(params["force_cross_attn"]["gate"].value)
        result["gate/abs_mean"] = jnp.mean(jnp.abs(gate))
        result["gate/abs_max"] = jnp.max(jnp.abs(gate))
    return result
