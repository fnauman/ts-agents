import numpy as np
import pytest

from src.cli.output import (
    extract_images_from_jsonable,
    extract_images_to_files,
    format_human,
    to_jsonable,
)
from src.core.base import DecompositionResult


def test_to_jsonable_numpy_array():
    arr = np.array([1.0, 2.0, 3.0])
    result = to_jsonable(arr)
    assert result == [1.0, 2.0, 3.0]


def test_to_jsonable_dataclass():
    result = DecompositionResult(
        method="stl",
        trend=np.array([1.0, 2.0]),
        seasonal=np.array([0.1, 0.2]),
        residual=np.array([0.0, 0.0]),
        period=2,
    )
    payload = to_jsonable(result)
    assert payload["method"] == "stl"
    assert payload["period"] == 2
    assert payload["trend"] == [1.0, 2.0]


def test_format_human_array():
    arr = np.array([1.0, 2.0, 3.0])
    output = format_human(arr)
    assert "Array:" in output


def test_extract_images_to_files_single_token(tmp_path):
    content = "Summary\n[IMAGE_DATA:ZmFrZQ==]\nDone"
    rewritten, paths = extract_images_to_files(
        content,
        image_dir=str(tmp_path / "assets"),
        filename_prefix="theta",
    )

    assert len(paths) == 1
    assert paths[0].name == "theta_01.png"
    assert paths[0].read_bytes() == b"fake"
    assert "[IMAGE_DATA:" not in rewritten
    assert f"[IMAGE_FILE:{paths[0]}]" in rewritten


def test_extract_images_to_files_multiple_tokens(tmp_path):
    content = "[IMAGE_DATA:Zmlyc3Q=]\n[IMAGE_DATA:c2Vjb25k]"
    rewritten, paths = extract_images_to_files(
        content,
        image_dir=str(tmp_path / "assets"),
        filename_prefix="forecast",
    )

    assert [path.name for path in paths] == ["forecast_01.png", "forecast_02.png"]
    assert paths[0].read_bytes() == b"first"
    assert paths[1].read_bytes() == b"second"
    assert rewritten.count("[IMAGE_FILE:") == 2


def test_extract_images_to_files_invalid_payload_raises(tmp_path):
    content = "[IMAGE_DATA:abc]"
    with pytest.raises(ValueError):
        extract_images_to_files(
            content,
            image_dir=str(tmp_path / "assets"),
            filename_prefix="broken",
        )


def test_extract_images_to_files_no_tokens_is_noop(tmp_path):
    content = "No images here."
    rewritten, paths = extract_images_to_files(
        content,
        image_dir=str(tmp_path / "assets"),
        filename_prefix="noop",
    )
    assert rewritten == content
    assert paths == []


def test_extract_images_from_jsonable_nested_payload(tmp_path):
    payload = {
        "message": "Top level [IMAGE_DATA:ZmFrZQ==]",
        "nested": ["a", {"details": "Inner [IMAGE_DATA:c2Vjb25k]"}],
    }
    rewritten, paths = extract_images_from_jsonable(
        payload,
        image_dir=str(tmp_path / "assets"),
        filename_prefix="json",
    )

    assert len(paths) == 2
    assert [path.name for path in paths] == ["json_01.png", "json_02.png"]
    assert "[IMAGE_DATA:" not in rewritten["message"]
    assert "[IMAGE_FILE:" in rewritten["message"]
    assert "[IMAGE_DATA:" not in rewritten["nested"][1]["details"]
    assert "[IMAGE_FILE:" in rewritten["nested"][1]["details"]
