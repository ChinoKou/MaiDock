from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from PIL import Image as PILImage

# 供应商对真实图片经常只声明通用二进制类型，缺失时引擎也会兜底成 application/octet-stream，
# 因此这些类型同样进入图片探测：Pillow 读不出图片头就会失败，代价只是一次文件头读取。
_GENERIC_BINARY_MEDIA_TYPES = frozenset({"", "application/octet-stream", "binary/octet-stream"})

# ISO BMFF（mp4）box 头是 4 字节 size + 4 字节类型；size 为 1 时改读随后的 64 位 largesize。
_BOX_HEADER_SIZE = 8
_LARGE_SIZE_MARKER = 1
# mvhd 负载最多读到 v1 布局的 duration 结束（32 字节）；按声明尺寸整读会让一个
# 谎报 box 大小的容器把整份产物拉进内存。
_MVHD_HEAD_SIZE = 32
# mvhd 用全 1 的 duration 表示"时长未知"，不能当成一个巨大的真实时长。
_UNKNOWN_DURATION_32 = 0xFFFFFFFF
_UNKNOWN_DURATION_64 = 0xFFFFFFFFFFFFFFFF


@dataclass(frozen=True, slots=True)
class ProbedMediaMetadata:
    """物化产物的可选几何与时长元数据，探测不到的字段保持 None。"""

    width: int | None = None
    height: int | None = None
    duration_seconds: float | None = None


def probe_media_metadata(path: Path, media_type: str) -> ProbedMediaMetadata:
    """
    探测已落盘产物的宽高与时长，任何失败都返回全 None 而不抛出。

    分发只看 media_type 的主类型，并忽略 `; charset=binary` 之类的参数：`image/*` 与通用二进制
    类型走图片探测，`video/*` 交给时长探测，其余类型（如 `text/plain`）不做任何读取。
    """

    essence = media_type.split(";", 1)[0].strip().lower()
    if essence.startswith("image/") or essence in _GENERIC_BINARY_MEDIA_TYPES:
        return _probe_image(path)
    if essence.startswith("video/"):
        return _probe_video(path)
    return ProbedMediaMetadata()


def _probe_image(path: Path) -> ProbedMediaMetadata:
    try:
        # 惰性打开只解析文件头，不解码像素，size 就已经可用。
        with PILImage.open(path) as image:
            width, height = image.size
    except Exception:
        # 元数据是纯增益信息：文件缺失、下载截断、根本不是图片、缺少可选编解码器或触发解压炸弹
        # 保护，都不能让一次已经成功的物化变成失败，所以这里必须宽泛吞掉并退回全 None。
        return ProbedMediaMetadata()
    return ProbedMediaMetadata(width=width, height=height)


def _probe_video(path: Path) -> ProbedMediaMetadata:
    try:
        duration = _probe_mp4_duration_seconds(path)
    except Exception:
        # 与图片探测同理：畸形容器、截断下载都只降级为"没有元数据"。
        return ProbedMediaMetadata()
    return ProbedMediaMetadata(duration_seconds=duration)


def _probe_mp4_duration_seconds(path: Path) -> float | None:
    """扫描 ISO BMFF 顶层 box 找 moov/mvhd 读时长，不引入任何解码依赖。

    只读 box 头与 mvhd 头部这几十个字节，不触碰媒体数据；结构对不上就返回 None。
    """

    with path.open("rb") as stream:
        for box_type, payload_start, payload_end in _iter_boxes(stream, path.stat().st_size):
            if box_type != b"moov":
                continue
            stream.seek(payload_start)
            for inner_type, inner_start, inner_end in _iter_boxes(stream, payload_end):
                if inner_type != b"mvhd":
                    continue
                stream.seek(inner_start)
                return _mvhd_duration_seconds(stream.read(min(inner_end - inner_start, _MVHD_HEAD_SIZE)))
    return None


def _iter_boxes(stream: BinaryIO, end: int) -> Iterator[tuple[bytes, int, int]]:
    """在 [当前位置, end) 内顺序产出 (box 类型, 负载起点, 负载终点)。

    size 为 1 表示紧随其后的是 64 位 largesize，为 0 表示该 box 延伸到区间末尾。
    任何越界或不足一个 box 头的残余都直接结束遍历。
    """

    while True:
        start = stream.tell()
        header = stream.read(_BOX_HEADER_SIZE)
        if len(header) < _BOX_HEADER_SIZE:
            return
        size = int.from_bytes(header[:4], "big")
        box_type = header[4:]
        payload_start = start + _BOX_HEADER_SIZE
        if size == _LARGE_SIZE_MARKER:
            largesize = stream.read(8)
            if len(largesize) < 8:
                return
            size = int.from_bytes(largesize, "big")
            payload_start = start + _BOX_HEADER_SIZE + 8
        elif size == 0:
            size = end - start
        box_end = start + size
        if box_end > end or payload_start > box_end:
            return
        yield box_type, payload_start, box_end
        stream.seek(box_end)


def _mvhd_duration_seconds(payload: bytes) -> float | None:
    if not payload:
        return None
    version = payload[0]
    if version == 0:
        if len(payload) < 20:
            return None
        timescale = int.from_bytes(payload[12:16], "big")
        duration = int.from_bytes(payload[16:20], "big")
        unknown = _UNKNOWN_DURATION_32
    elif version == 1:
        if len(payload) < 32:
            return None
        timescale = int.from_bytes(payload[20:24], "big")
        duration = int.from_bytes(payload[24:32], "big")
        unknown = _UNKNOWN_DURATION_64
    else:
        return None
    if timescale <= 0 or duration <= 0 or duration == unknown:
        return None
    return duration / timescale
