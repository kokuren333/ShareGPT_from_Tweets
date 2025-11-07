"""
Generate ShareGPT-style data by asking LM Studio to craft reverse questions for tweets.

Example:
    python generate_sharegpt_from_tweets.py ^
        --input plain_text_tweets.jsonl ^
        --output sharegpt_reverse_tweets.jsonl ^
        --model llama-3.1-70b-instruct ^
        --workers 4 --overwrite
"""

from __future__ import annotations

import argparse
import json
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import requests

DEFAULT_BASE_URL = "http://localhost:1234/v1"
DEFAULT_SYSTEM_PROMPT = (
    "You rewrite statements into plausible Japanese user questions. "
    "Always paraphrase the idea, avoid copying long fragments verbatim, and output exactly one question."
)
DEFAULT_USER_TEMPLATE = "\\n".join([
    '以下のツイート本文を読み、その内容を答えとして引き出すための自然な質問を日本語で1つだけ作成してください。',
    '',
    '制約:',
    '- ツイート本文の語句を必要以上に繰り返さず、要点を言い換える',
    '- 疑問文にする（語尾は「?」「？」など）。命令・提案の形にしない',
    '- 一般的な聞き手が状況を尋ねる体で、余計な前置きや引用を付けない',
    '- 出力は質問文のみ',
    '',
    'ツイート本文:',
    '"""{tweet}"""',
    '',
    '出力:'
])

@dataclass
class ShareGPTRecord:
    order: int
    data: Dict[str, Any]


class LMStudioClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: Optional[str],
        temperature: float,
        top_p: float,
        max_tokens: int,
        timeout: float,
        max_retries: int,
        system_prompt: str,
        user_template: str,
    ) -> None:
        self.endpoint = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.system_prompt = system_prompt
        self.user_template = user_template

    def _build_payload(self, tweet_text: str) -> Dict[str, Any]:
        user_prompt = self.user_template.format(tweet=tweet_text.strip())
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }

    def generate_question(self, tweet_text: str) -> str:
        payload = self._build_payload(tweet_text)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.post(
                    self.endpoint, headers=headers, json=payload, timeout=self.timeout
                )
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"].strip()
                if content:
                    return content
                raise ValueError("Model returned empty content.")
            except (requests.RequestException, ValueError, KeyError, IndexError) as exc:
                last_error = exc
                backoff = min(2**attempt, 30)
                print(f"[warn] LM call failed (attempt {attempt}/{self.max_retries}): {exc}. Retrying in {backoff}s")
                time.sleep(backoff)

        raise RuntimeError("Exceeded max retries when contacting LM Studio") from last_error


def iter_plain_text_records(
    input_path: Path, start_line: int, max_records: Optional[int]
) -> Iterable[Tuple[int, str]]:
    processed = 0
    with input_path.open("r", encoding="utf-8") as handle:
        for line_idx, line in enumerate(handle):
            if line_idx < start_line:
                continue
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            tweet_text = payload.get("full_text")
            if not tweet_text:
                continue
            yield line_idx, tweet_text
            processed += 1
            if max_records is not None and processed >= max_records:
                break


def worker_loop(
    worker_id: int,
    client: LMStudioClient,
    source_label: Optional[str],
    score: Optional[float],
    include_metadata: bool,
    task_queue: "queue.Queue[Optional[Tuple[int, int, str]]]",
    result_queue: "queue.Queue[Tuple[int, Optional[ShareGPTRecord], Optional[str]]]",
) -> None:
    while True:
        task = task_queue.get()
        if task is None:
            task_queue.task_done()
            break
        order, line_idx, tweet_text = task
        try:
            question = client.generate_question(tweet_text)
            conversations = [
                {"from": "human", "value": question},
                {"from": "gpt", "value": tweet_text},
            ]
            payload: Dict[str, Any] = {"conversations": conversations}
            if source_label:
                payload["source"] = source_label
            if score is not None:
                payload["score"] = score
            if include_metadata:
                payload["metadata"] = {"input_line": line_idx}
            record = ShareGPTRecord(order=order, data=payload)
            result_queue.put((order, record, None))
        except Exception as exc:  # pylint: disable=broad-except
            result_queue.put((order, None, f"worker-{worker_id}: {exc}"))
        finally:
            task_queue.task_done()


def writer_loop(
    output_path: Path,
    mode: str,
    result_queue: "queue.Queue[Tuple[int, Optional[ShareGPTRecord], Optional[str]]]",
    stop_token: object,
    log_every: int,
) -> None:
    pending: Dict[int, ShareGPTRecord] = {}
    next_order = 0
    processed = 0
    failures = 0
    with output_path.open(mode, encoding="utf-8") as handle:
        while True:
            item = result_queue.get()
            if item is stop_token:
                result_queue.task_done()
                break
            order, record, error = item
            if error:
                failures += 1
                print(f"[error] {error}")
                pending[order] = ShareGPTRecord(order=order, data={})
            elif record:
                pending[order] = record
            while next_order in pending:
                ready = pending.pop(next_order)
                if ready.data:
                    handle.write(json.dumps(ready.data, ensure_ascii=False) + "\n")
                    handle.flush()
                next_order += 1
            processed += 1
            if log_every and processed % log_every == 0:
                print(f"[info] writer processed {processed} items (failures={failures})")
            result_queue.task_done()

        if pending:
            raise RuntimeError(f"Writer exiting with {len(pending)} pending records.")
    print(f"[done] Wrote {processed - failures} rows (failures={failures}).")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use LM Studio to generate reverse questions and build a ShareGPT JSONL dataset."
    )
    parser.add_argument("--input", default="plain_text_tweets.jsonl", help="plain_text_tweets.jsonl produced earlier.")
    parser.add_argument("--output", default="sharegpt_reverse_tweets.jsonl", help="Destination ShareGPT JSONL path.")
    parser.add_argument("--model", default="openai/gpt-oss-20b", help="Model name as known to LM Studio.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="LM Studio OpenAI-compatible base URL.")
    parser.add_argument("--api-key", default=None, help="Optional API key if LM Studio enforces auth.")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p nucleus sampling.")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Max tokens for generated question.")
    parser.add_argument("--timeout", type=float, default=60.0, help="HTTP timeout per request (seconds).")
    parser.add_argument("--max-retries", type=int, default=5, help="Retry count for LM calls.")
    parser.add_argument("--workers", type=int, default=12, help="Number of concurrent LM workers.")
    parser.add_argument(
        "--prefetch-multiplier",
        type=int,
        default=4,
        help="Task queue capacity = workers * prefetch_multiplier.",
    )
    parser.add_argument("--start-line", type=int, default=0, help="Skip tweets before this zero-based line index.")
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Limit the number of tweets processed (for manual batching).",
    )
    parser.add_argument("--log-every", type=int, default=50, help="Writer progress interval (0 disables).")
    parser.add_argument("--source-label", default=None, help="Optional value for the ShareGPT 'source' field.")
    parser.add_argument("--score", type=float, default=None, help="Optional numeric score stored with each record.")
    parser.add_argument(
        "--include-metadata",
        action="store_true",
        help="Include an auxiliary metadata block (default: omit).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file if it exists (default: fail to avoid clobbering).",
    )
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="Custom system prompt for LM Studio.",
    )
    parser.add_argument(
        "--prompt-template",
        default=DEFAULT_USER_TEMPLATE,
        help="User prompt template. Must contain '{tweet}'.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} already exists. Use --overwrite to replace it.")
    if args.workers < 1:
        raise ValueError("workers must be >= 1")
    if args.prefetch_multiplier < 1:
        raise ValueError("prefetch-multiplier must be >= 1")

    client = LMStudioClient(
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        max_retries=args.max_retries,
        system_prompt=args.system_prompt,
        user_template=args.prompt_template,
    )

    task_queue: "queue.Queue[Optional[Tuple[int, int, str]]]" = queue.Queue(
        maxsize=max(1, args.workers * args.prefetch_multiplier)
    )
    result_queue: "queue.Queue[Tuple[int, Optional[ShareGPTRecord], Optional[str]]]" = queue.Queue()
    stop_token = object()

    writer_thread = threading.Thread(
        target=writer_loop,
        args=(
            output_path,
            "w" if args.overwrite else "x",
            result_queue,
            stop_token,
            args.log_every,
        ),
        daemon=True,
    )
    writer_thread.start()

    workers = []
    for worker_id in range(args.workers):
        thread = threading.Thread(
            target=worker_loop,
            kwargs={
                "worker_id": worker_id,
                "client": client,
                "source_label": args.source_label,
                "score": args.score,
                "include_metadata": args.include_metadata,
                "task_queue": task_queue,
                "result_queue": result_queue,
            },
            daemon=True,
        )
        thread.start()
        workers.append(thread)

    jobs_enqueued = 0
    for line_idx, tweet_text in iter_plain_text_records(input_path, args.start_line, args.max_records):
        task_queue.put((jobs_enqueued, line_idx, tweet_text))
        jobs_enqueued += 1

    if jobs_enqueued == 0:
        print("No tweets matched the criteria. Creating an empty output file.")

    for _ in workers:
        task_queue.put(None)

    task_queue.join()
    result_queue.put(stop_token)
    result_queue.join()

    for thread in workers:
        thread.join()

    writer_thread.join()
    print(f"Finished processing {jobs_enqueued} tweets into ShareGPT format at {output_path}")


if __name__ == "__main__":
    main()
