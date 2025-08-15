#!/usr/bin/env python3
import argparse
import io
import json
import logging
import os
from typing import Dict, List, Optional, Tuple, Any

from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq

SYSTEM_PROMPT = """You are a helpful assistant.

# Tools
You may call one or more functions to assist with the user query.
You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type":"function","function":{"name":"image_zoom_in_tool","description":"Zoom in on a specific region of an image by cropping it based on a bounding box (bbox) and an optional object label.","parameters":{"type":"object","properties":{"bbox_2d":{"type":"array","items":{"type":"number"},"minItems":4,"maxItems":4,"description":"The bounding box of the region to zoom in, as [x1, y1, x2, y2], where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner."},"label":{"type":"string","description":"The name or label of the object in the specified bounding box (optional)."}},"required":["bbox_2d"]}}}
</tools>

# How to call a tool
Return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

**Example**:  
<tool_call>  
{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [10, 20, 100, 200], "label": "the apple on the desk"}}  
</tool_call>"""

USER_PROMPT_TEMPLATE = (
    "<image>\n{question}\n"
    "Think first, call **image_zoom_in_tool** if needed, then answer. "
    "Format strictly as:  <think>...</think>  <tool_call>...</tool_call> (if tools needed)  <answer>...</answer> "
)

FIXED_ABILITY = "vl_chart"
FIXED_ENV_NAME = "visual_toolbox_v2"


def resolve_image_path(img_path: str, images_root: Optional[str], jsonl_dir: str) -> str:
    if os.path.isabs(img_path) and os.path.exists(img_path):
        return img_path
    if images_root:
        candidate = os.path.join(images_root, img_path)
        if os.path.exists(candidate):
            return candidate
    return os.path.join(jsonl_dir, img_path)


def load_image_bytes(path: str, reencode_format: str = "KEEP") -> bytes:
    with Image.open(path) as im:
        im.load()
        if reencode_format.upper() == "KEEP":
            fmt = im.format or "PNG"
        else:
            fmt = reencode_format.upper()
        if fmt == "JPEG" and im.mode in ("RGBA", "P"):
            im = im.convert("RGB")
        buf = io.BytesIO()
        save_kwargs = {}
        if fmt == "JPEG":
            save_kwargs.update(quality=95, optimize=True)
        im.save(buf, format=fmt, **save_kwargs)
        return buf.getvalue()


def extract_qa(conversations: List[Dict[str, Any]]) -> Tuple[str, str]:
    q, a = None, None
    for turn in conversations:
        frm = (turn.get("from") or "").lower()
        if frm == "human" and q is None:
            q = turn.get("value", "")
        elif frm == "gpt" and a is None:
            a = turn.get("value", "")
        if q is not None and a is not None:
            break
    if q is None and len(conversations) >= 1:
        q = conversations[0].get("value", "")
    if a is None and len(conversations) >= 2:
        a = conversations[1].get("value", "")
    return q or "", a or ""


def to_bytes(obj: Any) -> bytes:
    """Coerce various inputs to real bytes; raise with helpful context if not possible."""
    if isinstance(obj, (bytes, bytearray, memoryview)):
        return bytes(obj)
    try:
        import numpy as np  # optional
        if isinstance(obj, np.ndarray):
            return obj.tobytes()
    except Exception:
        pass
    if isinstance(obj, list) and all(isinstance(x, int) for x in obj):
        # allow a list of ints 0..255
        try:
            return bytes(obj)
        except ValueError as e:
            raise TypeError(f"Cannot convert list of ints to bytes: {e}")
    raise TypeError(f"Expected bytes-like object, got {type(obj).__name__}")


def build_schema() -> pa.schema:
    prompt_struct = pa.struct([
        pa.field("content", pa.string()),
        pa.field("role", pa.string()),
    ])
    image_struct = pa.struct([pa.field("bytes", pa.large_binary())])  # large_binary for robustness
    reward_struct = pa.struct([
        pa.field("ground_truth", pa.string()),
        pa.field("style", pa.string()),
    ])
    extra_struct = pa.struct([
        pa.field("answer", pa.string()),
        pa.field("index", pa.string()),
        pa.field("question", pa.string()),
        pa.field("split", pa.string()),
    ])
    return pa.schema([
        pa.field("data_source", pa.string()),
        pa.field("prompt", pa.list_(prompt_struct)),
        pa.field("images", pa.list_(image_struct)),
        pa.field("ability", pa.string()),
        pa.field("env_name", pa.string()),
        pa.field("reward_model", reward_struct),
        pa.field("extra_info", extra_struct),
    ])


def validate_record(row: Dict[str, Any], idx: int) -> None:
    """Raise a clear error if any field is malformed."""
    # images shape & type
    imgs = row.get("images")
    if not isinstance(imgs, list) or not imgs:
        raise ValueError(f"Row {idx}: 'images' must be a non-empty list; got {type(imgs).__name__}")
    for j, img in enumerate(imgs):
        if not isinstance(img, dict) or "bytes" not in img:
            raise ValueError(f"Row {idx}: images[{j}] must be a dict with a 'bytes' key")
        b = img["bytes"]
        try:
            img["bytes"] = to_bytes(b)  # coerce & ensure correct type
        except Exception as e:
            raise TypeError(f"Row {idx}: images[{j}]['bytes'] not bytes-like: {e}")
    # prompt fields
    pr = row.get("prompt")
    if not isinstance(pr, list) or len(pr) != 2:
        raise ValueError(f"Row {idx}: 'prompt' must be a 2-element list [system,user]")
    for j, p in enumerate(pr):
        if not isinstance(p, dict) or "content" not in p or "role" not in p:
            raise ValueError(f"Row {idx}: prompt[{j}] must have 'content' and 'role'")
        if not isinstance(p["content"], str) or not isinstance(p["role"], str):
            raise TypeError(f"Row {idx}: prompt[{j}] 'content' and 'role' must be strings")
    # reward_model
    rm = row.get("reward_model", {})
    if not isinstance(rm.get("ground_truth", ""), str) or rm.get("style") != "model":
        raise ValueError(f"Row {idx}: reward_model malformed: {rm}")
    # extra_info
    ei = row.get("extra_info", {})
    for key in ("answer", "index", "question", "split"):
        if not isinstance(ei.get(key, ""), str):
            raise TypeError(f"Row {idx}: extra_info['{key}'] must be str")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--data-source", required=True)
    ap.add_argument("--images-root", default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--reencode-format", default="KEEP", choices=["KEEP", "PNG", "JPEG"])
    ap.add_argument("--skip-missing", action="store_true")
    ap.add_argument("--debug", action="store_true", help="Print the first offending row on failure")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    jsonl_dir = os.path.dirname(os.path.abspath(args.input))
    schema = build_schema()
    records: List[Dict[str, Any]] = []

    total = 0
    kept = 0

    with open(args.input, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                logging.warning(f"Skipping malformed JSON on line {idx+1}: {e}")
                continue

            question, answer = extract_qa(item.get("conversations", []))

            prompt = [
                {"content": SYSTEM_PROMPT, "role": "system"},
                {"content": USER_PROMPT_TEMPLATE.format(question=question), "role": "user"},
            ]

            img_rel = item.get("image", "")
            img_path = resolve_image_path(img_rel, args.images_root, jsonl_dir)
            try:
                img_bytes = load_image_bytes(img_path, args.reencode_format)
            except Exception as e:
                msg = f"Failed to load image for line {idx+1} ('{img_rel}' -> '{img_path}'): {e}"
                if args.skip_missing:
                    logging.warning(msg + " [skipped]")
                    continue
                raise RuntimeError(msg) from e

            row = {
                "data_source": args.data_source,
                "prompt": prompt,
                "images": [{"bytes": img_bytes}],  # will be coerced to bytes explicitly in validate_record
                "ability": FIXED_ABILITY,
                "env_name": FIXED_ENV_NAME,
                "reward_model": {"ground_truth": str(answer), "style": "model"},
                "extra_info": {
                    "answer": str(answer),
                    "index": str(idx),
                    "question": question,
                    "split": args.split,
                },
            }

            # Validate & coerce before collecting
            validate_record(row, idx)
            records.append(row)
            kept += 1

    if not records:
        raise RuntimeError("No records to write. Check your input and filtering options.")

    try:
        table = pa.Table.from_pylist(records, schema=schema)
    except Exception as e:
        if args.debug:
            # Find & print the first bad row (re-validate with extra logging)
            for i, r in enumerate(records):
                try:
                    validate_record(r, i)
                except Exception as inner:
                    logging.error(f"First failing row appears to be index {i}: {inner}")
                    logging.error(f"Row {i} (truncated preview): " +
                                  json.dumps({k: (str(v)[:200] if k != "images" else f'images[{len(v)}] bytes')
                                              for k, v in r.items()}, ensure_ascii=False))
                    break
        raise

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    pq.write_table(table, args.output, compression="zstd")
    logging.info(f"Wrote {kept} rows to {args.output} (from {total} JSONL lines).")


if __name__ == "__main__":
    main()
