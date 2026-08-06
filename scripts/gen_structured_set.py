"""Generate 100 seeded JSON-schema-conditioned generation tasks.

Outputs to data/eval/structured/items.jsonl.
Each item: {id, schema, instruction, prompt_text}.
Deterministic: seed=42, no external data.
Schemas are our own construction (no license concerns).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

SEED = 42
NUM_ITEMS = 100
OUT_DIR = Path("data/eval/structured")

# ── Schema templates ──────────────────────────────────────────────────
# Each template is a (schema, instruction, prompt_stem) tuple.
# We generate 100 items by varying fields within 10 archetype schemas.

SCHEMA_ARCHETYPES: list[dict] = [
    # 0: person record
    {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer", "minimum": 0},
            "email": {"type": "string", "format": "email"},
        },
        "required": ["name", "age", "email"],
    },
    # 1: product listing
    {
        "type": "object",
        "properties": {
            "product_name": {"type": "string"},
            "price_usd": {"type": "number", "minimum": 0},
            "in_stock": {"type": "boolean"},
            "tags": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["product_name", "price_usd", "in_stock"],
    },
    # 2: calendar event
    {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "start_iso": {"type": "string", "format": "date-time"},
            "duration_minutes": {"type": "integer", "minimum": 1},
            "attendees": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["title", "start_iso", "duration_minutes"],
    },
    # 3: address
    {
        "type": "object",
        "properties": {
            "street": {"type": "string"},
            "city": {"type": "string"},
            "state": {"type": "string", "minLength": 2, "maxLength": 2},
            "zip": {"type": "string", "pattern": "^\\d{5}$"},
        },
        "required": ["street", "city", "state", "zip"],
    },
    # 4: recipe
    {
        "type": "object",
        "properties": {
            "dish": {"type": "string"},
            "servings": {"type": "integer", "minimum": 1},
            "ingredients": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "item": {"type": "string"},
                        "quantity": {"type": "string"},
                    },
                    "required": ["item", "quantity"],
                },
            },
        },
        "required": ["dish", "servings", "ingredients"],
    },
    # 5: weather report
    {
        "type": "object",
        "properties": {
            "location": {"type": "string"},
            "temperature_c": {"type": "number"},
            "conditions": {"type": "string", "enum": ["sunny", "cloudy", "rainy", "snowy", "windy"]},
            "humidity_pct": {"type": "integer", "minimum": 0, "maximum": 100},
        },
        "required": ["location", "temperature_c", "conditions"],
    },
    # 6: book reference
    {
        "type": "object",
        "properties": {
            "isbn": {"type": "string", "pattern": "^\\d{13}$"},
            "title": {"type": "string"},
            "author": {"type": "string"},
            "year": {"type": "integer", "minimum": 1400, "maximum": 2100},
        },
        "required": ["isbn", "title", "author", "year"],
    },
    # 7: API response
    {
        "type": "object",
        "properties": {
            "status": {"type": "integer", "enum": [200, 201, 400, 404, 500]},
            "data": {"type": "object"},
            "error_message": {"type": "string"},
        },
        "required": ["status"],
    },
    # 8: student record
    {
        "type": "object",
        "properties": {
            "student_id": {"type": "string", "pattern": "^[A-Z]{2}\\d{6}$"},
            "name": {"type": "string"},
            "gpa": {"type": "number", "minimum": 0.0, "maximum": 4.0},
            "enrolled_courses": {"type": "array", "items": {"type": "string"}, "maxItems": 6},
        },
        "required": ["student_id", "name", "gpa"],
    },
    # 9: measurement log
    {
        "type": "object",
        "properties": {
            "sensor_id": {"type": "string"},
            "timestamp_iso": {"type": "string", "format": "date-time"},
            "value": {"type": "number"},
            "unit": {"type": "string", "enum": ["celsius", "pascal", "lux", "dB"]},
        },
        "required": ["sensor_id", "timestamp_iso", "value", "unit"],
    },
]

INSTRUCTION_TEMPLATES = [
    "Generate a valid JSON object conforming to the provided schema about: {topic}.",
    "Create a JSON document matching the given schema for: {topic}.",
    "Produce a JSON object that satisfies the schema below. Context: {topic}.",
    "Write a well-formed JSON instance for the schema, scenario: {topic}.",
    "Emit a single JSON object adhering to the schema. Use-case: {topic}.",
]

TOPICS = [
    "a fictional employee profile",
    "a vintage vinyl record listing",
    "a space mission launch event",
    "a Tokyo apartment address",
    "a Thai curry recipe",
    "a tropical island weather report",
    "a classic science-fiction novel",
    "a REST API health-check response",
    "a graduate student transcript",
    "an industrial temperature sensor reading",
    "a freelance contractor profile",
    "a rare coffee bean product page",
    "a board meeting schedule",
    "a Berlin office address",
    "a sourdough bread recipe",
    "an arctic weather station report",
    "a medieval manuscript reference",
    "a payment gateway webhook response",
    "a high-school student record",
    "a seismograph station log",
]


def _build_prompt(schema: dict, instruction: str) -> str:
    schema_json = json.dumps(schema, indent=2, ensure_ascii=False)
    return f"{instruction}\n\nSchema:\n```json\n{schema_json}\n```"


def generate() -> None:
    rng = random.Random(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "items.jsonl"

    items: list[dict] = []
    for i in range(NUM_ITEMS):
        archetype_idx = i % len(SCHEMA_ARCHETYPES)
        schema = SCHEMA_ARCHETYPES[archetype_idx]
        topic = TOPICS[i % len(TOPICS)]
        instruction_tpl = INSTRUCTION_TEMPLATES[i % len(INSTRUCTION_TEMPLATES)]
        instruction = instruction_tpl.format(topic=topic)
        prompt_text = _build_prompt(schema, instruction)

        items.append({
            "id": f"structured-{i:04d}",
            "schema": schema,
            "instruction": instruction,
            "prompt_text": prompt_text,
        })

    # Shuffle deterministically for variety in ordering
    rng.shuffle(items)

    with out_path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"gen_structured_set: wrote {len(items)} items to {out_path}")


if __name__ == "__main__":
    generate()
