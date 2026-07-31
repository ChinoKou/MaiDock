from collections.abc import Mapping

from ...core.json_types import JsonValue

# 插件与 Host 之间的 RPC 边界：请求体由 SDK 反序列化后传进来，响应体由我们构造后交回去，
# 两侧都只可能是纯 JSON。入参用 Mapping 表达"只读、不持有、不修改"，返回用 dict 表达
# "新建的、调用方可以随意处置"。六个 adapter 与 runtime 的 Protocol/ingress 共用这两个别名，
# 保证边界两端的形状由一处定义，而不是二十四处各写一遍裸 dict。
type HostRpcRequest = Mapping[str, JsonValue]
type HostRpcResponse = dict[str, JsonValue]
