"""供应商无关的模型目录约束原语。

每个供应商的 registry 都要回答同两个问题：这个 mode 需要哪些输入角色、
这个参数的取值合不合法。约束的*表达方式*与供应商无关，只有约束的*内容*
（模型目录）才是供应商特有的，因此把这两个原语放在 providers/common，
避免第二家供应商照抄一遍。
"""

from dataclasses import dataclass

from ...domain import MediaInputRole


@dataclass(frozen=True, slots=True)
class ParameterConstraint:
    kinds: tuple[type, ...]
    minimum: float | None = None
    maximum: float | None = None
    choices: frozenset[object] = frozenset()

    def validate(self, name: str, value: object) -> None:
        if not isinstance(value, self.kinds) or isinstance(value, bool) and bool not in self.kinds:
            expected = "/".join(kind.__name__ for kind in self.kinds)
            raise ValueError(f"参数 {name} 必须是 {expected}")
        if self.choices and value not in self.choices:
            allowed = ", ".join(str(item) for item in sorted(self.choices, key=str))
            raise ValueError(f"参数 {name} 只允许: {allowed}")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if self.minimum is not None and value < self.minimum:
                raise ValueError(f"参数 {name} 不能小于 {self.minimum:g}")
            if self.maximum is not None and value > self.maximum:
                raise ValueError(f"参数 {name} 不能大于 {self.maximum:g}")


@dataclass(frozen=True, slots=True)
class ModeConstraint:
    required_roles: frozenset[MediaInputRole] = frozenset()
    allowed_roles: frozenset[MediaInputRole] = frozenset()
    prompt_required: bool = False
