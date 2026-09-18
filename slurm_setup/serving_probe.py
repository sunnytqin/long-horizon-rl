"""Dependency-free preflight and bounded OpenAI-compatible serving smoke.

This is a transport/format smoke, NOT an assessment of simulator/judge quality.
"""

import argparse
import concurrent.futures
import json
from pathlib import Path
import time
import urllib.request


def check_weights(directory):
    root = Path(directory)
    config = json.loads((root / "config.json").read_text())
    index = root / "model.safetensors.index.json"
    if index.is_file():
        shards = sorted(set(json.loads(index.read_text())["weight_map"].values()))
    else:
        shards = ["model.safetensors"]
    if not shards:
        raise ValueError("empty weight index")
    for shard in shards:
        path = root / shard
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"missing/empty weight shard: {path}")
    if not (root / "tokenizer_config.json").is_file():
        raise ValueError(f"missing tokenizer_config.json: {root}")
    # Summary only: a block-FP8 config carries a several-hundred-entry
    # modules_to_not_convert list that would bury the launcher's output.
    quant = config.get("quantization_config") or {}
    summary = {k: quant[k] for k in ("quant_method", "fmt", "weight_block_size")
               if k in quant} or None
    print(json.dumps({"weights": str(root), "shards": len(shards),
                      "quantization": summary}))


def chat_probe(base_url, model, out):
    cases = [
        ("chat", "Reply with exactly the word READY."),
        ("sim", "Role-play a customer asking for a function that sorts integers. "
         "Reply to: 'Should the order be ascending or descending?' "
         "Say ascending in natural language; do not write code."),
        ("json", 'Classify this user reply: "Please sort in ascending order." '
         'Return only a JSON object with one boolean field "contains_code".'),
    ]

    def request(case):
        label, prompt = case
        payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
                   "max_tokens": 256, "temperature": 0}
        start = time.monotonic()
        req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions",
                                     data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json",
                                              "Authorization": "Bearer EMPTY"})
        with urllib.request.urlopen(req, timeout=180) as response:
            result = json.load(response)
        choice = result["choices"][0]
        content = choice["message"]["content"]
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"{label}: empty chat content")
        if choice.get("finish_reason") == "length":
            raise ValueError(f"{label}: truncated response")
        return {"case": label, "seconds": time.monotonic() - start, "response": result}

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(request, cases))
    # Save raw responses even when the following format assertion fails.
    Path(out).write_text(json.dumps({"model": model, "base_url": base_url,
                                     "requests": records}, indent=2) + "\n")
    value = json.loads(records[2]["response"]["choices"][0]["message"]["content"])
    if not isinstance(value, dict) or type(value.get("contains_code")) is not bool:
        raise ValueError("judge-shaped response is not the requested JSON schema")
    print(f"PASS: three concurrent completions, including JSON output; saved {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("weights").add_argument("directory")
    chat = sub.add_parser("chat")
    chat.add_argument("--base_url", required=True)
    chat.add_argument("--model", required=True)
    chat.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.command == "weights":
        check_weights(args.directory)
    else:
        chat_probe(args.base_url, args.model, args.out)


if __name__ == "__main__":
    main()
