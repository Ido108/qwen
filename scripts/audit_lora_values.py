"""Read installed LoRA tensors on CPU. Do not load models or generate images."""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from safetensors import safe_open
from backend.models import catalog, local_path

torch.set_num_threads(2)
rows = []
for item in catalog():
    if "lora" not in item.get("kind", "") or not item.get("files"):
        continue
    path = local_path(item, item["files"][0])
    if not path.is_file() or path.suffix != ".safetensors":
        continue
    row = {"id": item["id"], "title": item["title"], "groups": {}, "zero_up": [], "nonfinite": []}
    groups = defaultdict(lambda: {"pairs": 0, "zero_up": 0, "up_squared_norm": 0., "alpha_over_rank": []})
    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        row["metadata"] = handle.metadata()
        for key in sorted(keys):
            suffix = next((s for s in (".lora_B.weight", ".lora_up.weight") if key.endswith(s)), None)
            if not suffix:
                continue
            root = key[:-len(suffix)]
            group = "mlp" if "mlp" in key or ".ff." in key else "attention" if "attn" in key else "other"
            values = handle.get_tensor(key).float()
            data = groups[group]
            data["pairs"] += 1
            zero = not bool(torch.count_nonzero(values))
            data["zero_up"] += int(zero)
            data["up_squared_norm"] += float(values.square().sum())
            if zero:
                row["zero_up"].append(key)
            if not bool(torch.isfinite(values).all()):
                row["nonfinite"].append(key)
            rank = values.shape[1]
            alpha = float(handle.get_tensor(root + ".alpha")) if root + ".alpha" in keys else rank
            if alpha / rank not in data["alpha_over_rank"]:
                data["alpha_over_rank"].append(alpha / rank)
    row["groups"] = dict(groups)
    rows.append(row)
target = Path(sys.argv[1])
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(json.dumps(rows, ensure_ascii=True, indent=2), encoding="utf-8")
print(json.dumps([{k: r[k] for k in ("id", "groups", "nonfinite")} for r in rows]))
