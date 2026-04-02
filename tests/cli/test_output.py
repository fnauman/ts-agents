import numpy as np
import pytest

from ts_agents.contracts import ArtifactRef, ToolPayload
from ts_agents.cli.output import (
    dump_json,
    extract_images_from_jsonable,
    extract_images_to_files,
    format_human,
    to_jsonable,
)
from ts_agents.core.base import DecompositionResult


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


def test_to_jsonable_sanitizes_non_finite_floats():
    payload = to_jsonable({"nan": float("nan"), "inf": np.float64("inf"), "ok": 1.5})
    assert payload == {"nan": None, "inf": None, "ok": 1.5}


def test_dump_json_emits_strict_json_for_non_finite_values():
    rendered = dump_json({"scores": {"32": float("nan"), "64": 0.5}})
    assert '"32": null' in rendered
    assert "NaN" not in rendered


def test_to_jsonable_raises_for_unsupported_objects():
    with pytest.raises(TypeError):
        to_jsonable(object())


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


def test_format_human_tool_payload_includes_artifacts():
    payload = ToolPayload(
        kind="patterns",
        summary="Detected 2 peaks.",
        data={"count": 2},
        artifacts=[
            ArtifactRef(
                kind="image",
                path="/tmp/peaks.png",
                mime_type="image/png",
                description="Peak detection plot",
            )
        ],
        warnings=["Review prominence threshold."],
    )

    output = format_human(payload)

    assert "Detected 2 peaks." in output
    assert "Warnings:" in output
    assert "Review prominence threshold." in output
    assert "Artifacts:" in output
    assert "/tmp/peaks.png" in output
    assert "Data:" in output
    assert "- count: 2" in output
