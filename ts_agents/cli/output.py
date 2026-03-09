"""Output helpers for CLI rendering and serialization."""

from __future__ import annotations

import base64
import binascii
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
import re
from typing import Any, List, Optional, Tuple

import numpy as np

IMAGE_DATA_PATTERN = re.compile(r"\[IMAGE_DATA:([A-Za-z0-9+/=]+)\]")


def to_jsonable(value: Any) -> Any:
    """Convert Python objects to JSON-serializable structures."""
    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, (np.integer, np.floating)):
        return value.item()

    if isinstance(value, np.ndarray):
        return value.tolist()

    if hasattr(value, "to_dict") and callable(value.to_dict):
        return to_jsonable(value.to_dict())

    if is_dataclass(value):
        return {k: to_jsonable(v) for k, v in asdict(value).items()}

    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]

    if hasattr(value, "__dict__"):
        return {k: to_jsonable(v) for k, v in value.__dict__.items() if not k.startswith("_")}

    return value


def format_human(value: Any) -> str:
    """Render a human-readable summary of a result."""
    if value is None:
        return "(no result)"

    if isinstance(value, str):
        return value

    if isinstance(value, (int, float, bool)):
        return str(value)

    if isinstance(value, np.ndarray):
        if value.size <= 10:
            return f"Array: {value.tolist()}"
        return (
            f"Array shape={value.shape}, min={value.min():.6g}, "
            f"max={value.max():.6g}"
        )

    if hasattr(value, "to_dict") and callable(value.to_dict):
        return format_human(value.to_dict())

    if is_dataclass(value):
        return format_human(asdict(value))

    if isinstance(value, dict):
        lines = []
        for key, item in value.items():
            if isinstance(item, (list, tuple)) and len(item) > 5:
                rendered = f"list[{len(item)}]"
            elif isinstance(item, dict):
                rendered = f"dict[{len(item)}]"
            elif isinstance(item, np.ndarray):
                rendered = f"array[{item.size}]"
            else:
                rendered = str(item)
            lines.append(f"- {key}: {rendered}")
        return "\n".join(lines)

    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return "[]"
        if len(value) > 10:
            return f"list[{len(value)}]"
        return "[" + ", ".join(str(v) for v in value) + "]"

    if hasattr(value, "__dict__"):
        return format_human({k: v for k, v in value.__dict__.items() if not k.startswith("_")})

    return str(value)


def render_output(
    result: Any,
    *,
    json_output: bool = False,
    text_output: Optional[str] = None,
) -> str:
    """Render output for printing or saving."""
    if json_output:
        payload = to_jsonable(result)
        return json.dumps(payload, indent=2, default=str)

    if text_output is not None:
        return text_output

    return format_human(result)


def write_output(content: str, path: str) -> Path:
    """Write output to disk and return the resolved path."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content)
    return output_path


def _decode_image_payload(raw_payload: str, index: int) -> bytes:
    try:
        return base64.b64decode(raw_payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"Invalid IMAGE_DATA payload at index {index}: {exc}") from exc


def _write_extracted_image(
    *,
    image_bytes: bytes,
    image_dir: Path,
    filename_prefix: str,
    index: int,
) -> Path:
    image_dir.mkdir(parents=True, exist_ok=True)
    image_path = image_dir / f"{filename_prefix}_{index:02d}.png"
    image_path.write_bytes(image_bytes)
    return image_path


def extract_images_to_files(
    content: str,
    *,
    image_dir: str,
    filename_prefix: str = "image",
) -> Tuple[str, List[Path]]:
    """Extract embedded image payloads and return rewritten content + file paths."""
    matches = list(IMAGE_DATA_PATTERN.finditer(content))
    if not matches:
        return content, []

    target_dir = Path(image_dir)
    image_paths: List[Path] = []

    def _replace(match: re.Match[str]) -> str:
        index = len(image_paths) + 1
        raw_payload = match.group(1)

        image_bytes = _decode_image_payload(raw_payload, index)
        image_path = _write_extracted_image(
            image_bytes=image_bytes,
            image_dir=target_dir,
            filename_prefix=filename_prefix,
            index=index,
        )
        image_paths.append(image_path)
        return f"[IMAGE_FILE:{image_path}]"

    rewritten = IMAGE_DATA_PATTERN.sub(_replace, content)
    return rewritten, image_paths


def extract_images_from_jsonable(
    value: Any,
    *,
    image_dir: str,
    filename_prefix: str = "image",
) -> Tuple[Any, List[Path]]:
    """Extract embedded image tokens from JSON-serializable objects."""
    target_dir = Path(image_dir)
    image_paths: List[Path] = []

    def _replace_tokens(text: str) -> str:
        def _replace(match: re.Match[str]) -> str:
            index = len(image_paths) + 1
            raw_payload = match.group(1)
            image_bytes = _decode_image_payload(raw_payload, index)
            image_path = _write_extracted_image(
                image_bytes=image_bytes,
                image_dir=target_dir,
                filename_prefix=filename_prefix,
                index=index,
            )
            image_paths.append(image_path)
            return f"[IMAGE_FILE:{image_path}]"

        return IMAGE_DATA_PATTERN.sub(_replace, text)

    def _walk(obj: Any) -> Any:
        if isinstance(obj, str):
            return _replace_tokens(obj)
        if isinstance(obj, list):
            return [_walk(item) for item in obj]
        if isinstance(obj, tuple):
            return [_walk(item) for item in obj]
        if isinstance(obj, dict):
            return {key: _walk(item) for key, item in obj.items()}
        return obj

    rewritten = _walk(value)
    return rewritten, image_paths
