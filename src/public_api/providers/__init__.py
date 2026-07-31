"""Public API 内部供应商贡献。"""

from ..catalog import PublicProviderContribution
from .dashscope.contribution import DASHSCOPE_PUBLIC_CONTRIBUTION
from .volcengine_ark.contribution import ARK_PUBLIC_CONTRIBUTION

PUBLIC_API_CONFIG_CATALOG: tuple[PublicProviderContribution, ...] = (
    DASHSCOPE_PUBLIC_CONTRIBUTION,
    ARK_PUBLIC_CONTRIBUTION,
)
PUBLIC_PROVIDER_CONTRIBUTIONS: tuple[PublicProviderContribution, ...] = (
    DASHSCOPE_PUBLIC_CONTRIBUTION,
    ARK_PUBLIC_CONTRIBUTION,
)

__all__ = ["PUBLIC_API_CONFIG_CATALOG", "PUBLIC_PROVIDER_CONTRIBUTIONS"]
