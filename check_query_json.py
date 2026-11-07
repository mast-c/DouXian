# -*- coding: utf-8 -*-
"""Isolate the EXACT query-to-JSON API call from the GraphRAG/Streamlit app.

Run from the medical Python project directory:
  python check_query_json.py --env-file .env

Does not connect to Neo4j or Chroma. Uses a public, non-personal test question,
prints no raw model output, question, endpoint host, API Key or reasoning text.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import urllib.error


QUESTION = "布地奈德鼻喷雾剂能否与含伪麻黄碱的复方感冒药同时使用？为什么？"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    args = parser.parse_args()
    os.environ["MEDICAL_QUERY_ENV_FILE"] = str(args.env_file.resolve())

    from medical_llm_query_understanding import (
        load_query_understanding_model_from_env,
        _model_invoke, _parse_response, _safe_response_meta
    )

    model = load_query_understanding_model_from_env()
    if model is None:
        print("CONFIG_ERROR: No complete API configuration found.")
        return 2
    print("Test: direct query-understanding API (not injected Streamlit model)")
    print("Configured model ID:", model.model_name)
    for attempt in range(2):
        try:
            response = _model_invoke(model, QUESTION, retry=bool(attempt))
            meta = _safe_response_meta(response)
            print("Attempt", attempt + 1, "metadata:",
                  json.dumps(meta, ensure_ascii=False, default=str))
            plan = _parse_response(response)
            print("SUCCESS: valid query JSON with",
                  len(plan.get("intents") or []), "intents and",
                  len(plan.get("entities") or []), "entities")
            return 0
        except urllib.error.HTTPError as exc:
            print("HTTP_ERROR:", exc.code, "check the model/region/parameters.")
            return 3
        except ValueError as exc:
            # Only known parser error labels; no raw model response.
            print("JSON_PARSE_ERROR: no usable JSON plan in this attempt.")
            if attempt == 1:
                return 4
        except Exception as exc:
            print("API_ERROR:", type(exc).__name__)
            return 5
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
