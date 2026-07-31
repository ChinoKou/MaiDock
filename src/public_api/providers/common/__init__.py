"""供应商包之间唯一允许的共享层：只放与具体供应商无关的原语。"""

from .constraints import ModeConstraint, ParameterConstraint

__all__ = ["ModeConstraint", "ParameterConstraint"]
