from pathlib import Path

from PIL import Image as PILImage

from src.public_api.application.media_probe import ProbedMediaMetadata, probe_media_metadata

_NOT_AN_IMAGE = b"\x00\x01\x02maidock-not-an-image\xff\xfe"

# mvhd 在 timescale/duration 之后还有 rate、volume、变换矩阵、next_track_id 等固定字段，
# 这里用零字节补齐，确保解析器只按偏移取值而不是依赖 payload 刚好结束。
_MVHD_TAIL = bytes(80)


def _write_image(path: Path, *, size: tuple[int, int], image_format: str) -> Path:
    with PILImage.new("RGB", size, color=(12, 34, 56)) as image:
        image.save(path, format=image_format)
    return path


def _box(box_type: bytes, payload: bytes) -> bytes:
    return (len(payload) + 8).to_bytes(4, "big") + box_type + payload


def _box_64bit(box_type: bytes, payload: bytes) -> bytes:
    """size 字段写 1、真实长度放进随后的 64 位 largesize，即 ISO BMFF 的大 box 形式。"""

    return (1).to_bytes(4, "big") + box_type + (len(payload) + 16).to_bytes(8, "big") + payload


def _mvhd_v0(*, timescale: int, duration: int) -> bytes:
    payload = (
        b"\x00\x00\x00\x00"  # version 0 + flags
        + bytes(4)  # creation_time
        + bytes(4)  # modification_time
        + timescale.to_bytes(4, "big")
        + duration.to_bytes(4, "big")
        + _MVHD_TAIL
    )
    return _box(b"mvhd", payload)


def _mvhd_v1(*, timescale: int, duration: int) -> bytes:
    payload = (
        b"\x01\x00\x00\x00"  # version 1 + flags
        + bytes(8)  # creation_time
        + bytes(8)  # modification_time
        + timescale.to_bytes(4, "big")
        + duration.to_bytes(8, "big")
        + _MVHD_TAIL
    )
    return _box(b"mvhd", payload)


def _write_mp4(path: Path, *, moov_payload: bytes, large_moov: bool = False) -> Path:
    """写一个只含 ftyp/free/moov 的最小 mp4：真实文件里 moov 前后总有别的 box。"""

    moov = _box_64bit(b"moov", moov_payload) if large_moov else _box(b"moov", moov_payload)
    path.write_bytes(_box(b"ftyp", b"isom\x00\x00\x02\x00mp41") + _box(b"free", bytes(16)) + moov)
    return path


def test_probe_reads_exact_png_dimensions(tmp_path: Path) -> None:
    path = _write_image(tmp_path / "sample.png", size=(7, 13), image_format="PNG")

    assert probe_media_metadata(path, "image/png") == ProbedMediaMetadata(width=7, height=13)


def test_probe_reads_exact_jpeg_dimensions(tmp_path: Path) -> None:
    path = _write_image(tmp_path / "sample.jpg", size=(21, 9), image_format="JPEG")

    assert probe_media_metadata(path, "image/jpeg") == ProbedMediaMetadata(width=21, height=9)


def test_probe_ignores_media_type_parameters(tmp_path: Path) -> None:
    path = _write_image(tmp_path / "sample.png", size=(4, 6), image_format="PNG")

    assert probe_media_metadata(path, "IMAGE/PNG; charset=binary") == ProbedMediaMetadata(width=4, height=6)


def test_probe_still_reads_image_declared_as_generic_binary(tmp_path: Path) -> None:
    path = _write_image(tmp_path / "sample.png", size=(5, 11), image_format="PNG")

    assert probe_media_metadata(path, "application/octet-stream") == ProbedMediaMetadata(width=5, height=11)


def test_probe_returns_empty_for_non_image_bytes(tmp_path: Path) -> None:
    path = tmp_path / "sample.bin"
    path.write_bytes(_NOT_AN_IMAGE)

    assert probe_media_metadata(path, "application/octet-stream") == ProbedMediaMetadata()


def test_probe_returns_empty_for_missing_path(tmp_path: Path) -> None:
    assert probe_media_metadata(tmp_path / "missing.png", "image/png") == ProbedMediaMetadata()


def test_probe_skips_text_media_type(tmp_path: Path) -> None:
    path = _write_image(tmp_path / "sample.png", size=(3, 3), image_format="PNG")

    assert probe_media_metadata(path, "text/plain; charset=utf-8") == ProbedMediaMetadata()


def test_probe_reads_mp4_duration_from_v0_mvhd(tmp_path: Path) -> None:
    path = _write_mp4(tmp_path / "sample.mp4", moov_payload=_mvhd_v0(timescale=600, duration=3600))

    assert probe_media_metadata(path, "video/mp4") == ProbedMediaMetadata(duration_seconds=6.0)


def test_probe_reads_mp4_duration_from_v1_mvhd(tmp_path: Path) -> None:
    path = _write_mp4(tmp_path / "sample.mp4", moov_payload=_mvhd_v1(timescale=1000, duration=10500))

    assert probe_media_metadata(path, "video/mp4") == ProbedMediaMetadata(duration_seconds=10.5)


def test_probe_reads_mp4_duration_through_64bit_moov_box(tmp_path: Path) -> None:
    path = _write_mp4(
        tmp_path / "sample.mp4",
        moov_payload=_mvhd_v0(timescale=90000, duration=180000),
        large_moov=True,
    )

    assert probe_media_metadata(path, "video/mp4") == ProbedMediaMetadata(duration_seconds=2.0)


def test_probe_skips_boxes_before_mvhd_inside_moov(tmp_path: Path) -> None:
    path = _write_mp4(
        tmp_path / "sample.mp4",
        moov_payload=_box(b"skip", bytes(24)) + _mvhd_v0(timescale=25, duration=125),
    )

    assert probe_media_metadata(path, "video/mp4") == ProbedMediaMetadata(duration_seconds=5.0)


def test_probe_rejects_zero_timescale(tmp_path: Path) -> None:
    path = _write_mp4(tmp_path / "sample.mp4", moov_payload=_mvhd_v0(timescale=0, duration=3600))

    assert probe_media_metadata(path, "video/mp4") == ProbedMediaMetadata()


def test_probe_rejects_unknown_duration_sentinel(tmp_path: Path) -> None:
    path = _write_mp4(tmp_path / "sample.mp4", moov_payload=_mvhd_v0(timescale=600, duration=0xFFFFFFFF))

    assert probe_media_metadata(path, "video/mp4") == ProbedMediaMetadata()


def test_probe_rejects_unsupported_mvhd_version(tmp_path: Path) -> None:
    payload = b"\x07\x00\x00\x00" + bytes(64)
    path = _write_mp4(tmp_path / "sample.mp4", moov_payload=_box(b"mvhd", payload))

    assert probe_media_metadata(path, "video/mp4") == ProbedMediaMetadata()


def test_probe_returns_empty_for_truncated_mvhd(tmp_path: Path) -> None:
    full = _write_mp4(tmp_path / "full.mp4", moov_payload=_mvhd_v0(timescale=600, duration=3600))
    path = tmp_path / "truncated.mp4"
    # 砍掉 mvhd 尾部：box 头声称的长度越过文件末尾，遍历必须停下而不是读出垃圾时长。
    path.write_bytes(full.read_bytes()[:-40])

    assert probe_media_metadata(path, "video/mp4") == ProbedMediaMetadata()


def test_probe_reads_mp4_duration_from_trailing_moov_with_implicit_size(tmp_path: Path) -> None:
    """size=0 表示该 box 一直延伸到文件末尾，是流式封装里 moov 常见的写法。"""

    mvhd = _mvhd_v0(timescale=48000, duration=96000)
    path = tmp_path / "sample.mp4"
    path.write_bytes(_box(b"ftyp", b"isom") + bytes(4) + b"moov" + mvhd)

    assert probe_media_metadata(path, "video/mp4") == ProbedMediaMetadata(duration_seconds=2.0)


def test_probe_returns_empty_for_moov_without_mvhd(tmp_path: Path) -> None:
    path = _write_mp4(tmp_path / "sample.mp4", moov_payload=_box(b"trak", bytes(32)))

    assert probe_media_metadata(path, "video/mp4") == ProbedMediaMetadata()


def test_probe_returns_empty_for_mvhd_payload_shorter_than_header(tmp_path: Path) -> None:
    # mvhd 自称 version 0 却只带了 12 字节负载，timescale/duration 根本不在文件里。
    path = _write_mp4(tmp_path / "sample.mp4", moov_payload=_box(b"mvhd", b"\x00\x00\x00\x00" + bytes(8)))

    assert probe_media_metadata(path, "video/mp4") == ProbedMediaMetadata()


def test_probe_returns_empty_for_mp4_without_moov(tmp_path: Path) -> None:
    path = tmp_path / "sample.mp4"
    path.write_bytes(_box(b"ftyp", b"isom") + _box(b"mdat", bytes(32)))

    assert probe_media_metadata(path, "video/mp4") == ProbedMediaMetadata()


def test_probe_returns_empty_for_non_media_bytes_declared_as_video(tmp_path: Path) -> None:
    path = tmp_path / "sample.mp4"
    path.write_bytes(_NOT_AN_IMAGE)

    assert probe_media_metadata(path, "video/mp4") == ProbedMediaMetadata()


def test_probe_returns_empty_for_missing_video_path(tmp_path: Path) -> None:
    assert probe_media_metadata(tmp_path / "missing.mp4", "video/mp4") == ProbedMediaMetadata()
