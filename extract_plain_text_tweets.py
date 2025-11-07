"""
Extract full-text tweets that contain only plain text from a Twitter archive tweet.js file.

Usage:
    python extract_plain_text_tweets.py --input tweets.js --output plain_text_tweets.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

LINK_PATTERN = re.compile(r"(https?://\S+|t\.co/\S+)", re.IGNORECASE)
MENTION_PATTERN = re.compile(r"(?<!\w)@[A-Za-z0-9_]+")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract plain-text tweets (no links, no quote retweets, no media) "
            "from a tweet.js archive and save them as JSON Lines."
        )
    )
    parser.add_argument(
        "-i",
        "--input",
        default="tweets.js",
        help="Path to the tweet.js file exported from Twitter (default: %(default)s).",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="plain_text_tweets.jsonl",
        help="Destination JSONL file for the filtered tweets (default: %(default)s).",
    )
    args = parser.parse_args()

    tweets = load_tweet_archive(Path(args.input))
    output_path = Path(args.output)

    total = 0
    kept = 0

    with output_path.open("w", encoding="utf-8") as out_file:
        for entry in tweets:
            tweet = entry.get("tweet", entry)
            total += 1
            full_text = get_full_text(tweet)
            if not full_text:
                continue
            if not is_plain_text_tweet(tweet, full_text):
                continue

            json.dump({"full_text": full_text}, out_file, ensure_ascii=False)
            out_file.write("\n")
            kept += 1

    print(f"Examined {total} tweets, wrote {kept} plain-text tweets to {output_path}")


def load_tweet_archive(path: Path) -> List[Dict[str, Any]]:
    """
    Twitter exports tweet data inside a JS assignment (window.YTD...). Strip the
    prefix/suffix so we can decode the JSON payload.
    """
    raw = path.read_text(encoding="utf-8")
    start = raw.find("[")
    end = raw.rfind("]")
    if start == -1 or end == -1:
        raise ValueError("Input file does not appear to contain a JSON array.")
    json_blob = raw[start : end + 1]
    return json.loads(json_blob)


def get_full_text(tweet: Dict[str, Any]) -> str:
    note_tweet = tweet.get("note_tweet") or {}
    note_result = (
        note_tweet.get("note_tweet_results", {})
        .get("result", {})
        .get("text")
    )
    if note_result:
        return note_result
    return tweet.get("full_text") or tweet.get("text") or ""


def is_plain_text_tweet(tweet: Dict[str, Any], text: str) -> bool:
    if tweet.get("is_quote_status"):
        return False
    if any(
        key in tweet
        for key in (
            "quoted_status",
            "quoted_status_id",
            "quoted_status_id_str",
        )
    ):
        return False
    if has_media(tweet):
        return False
    if has_links(tweet, text):
        return False
    if has_mentions(tweet, text):
        return False
    return True


def has_links(tweet: Dict[str, Any], text: str) -> bool:
    entities = tweet.get("entities") or {}
    if entities.get("urls"):
        return True
    note_urls = _note_entity_list(tweet, "urls")
    if note_urls:
        return True
    return bool(LINK_PATTERN.search(text))


def has_media(tweet: Dict[str, Any]) -> bool:
    entities = tweet.get("entities") or {}
    if entities.get("media"):
        return True
    if tweet.get("extended_entities"):
        return True
    if tweet.get("attachments"):
        return True
    if tweet.get("card"):
        return True
    note_media = _note_entity_list(tweet, "media") or _note_entity_list(tweet, "media_entities")
    return bool(note_media)


def has_mentions(tweet: Dict[str, Any], text: str) -> bool:
    entities = tweet.get("entities") or {}
    if entities.get("user_mentions"):
        return True
    note_mentions = _note_entity_list(tweet, "user_mentions")
    if note_mentions:
        return True
    return bool(MENTION_PATTERN.search(text))


def _note_entity_list(tweet: Dict[str, Any], key: str) -> Iterable[Dict[str, Any]]:
    note_tweet = tweet.get("note_tweet")
    if not note_tweet:
        return []
    result = note_tweet.get("note_tweet_results", {}).get("result", {})
    entity_set = result.get("entity_set", {})
    return entity_set.get(key) or []


if __name__ == "__main__":
    main()
