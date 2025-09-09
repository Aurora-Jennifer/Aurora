import json
import sys

from jsonschema import ValidationError, validate


def main(
    meta_path: str = "reports/canary_run.meta.json", schema_path: str = "reports/canary.schema.json"
) -> int:
    with open(meta_path) as f:
        data = json.loads(f.read())
    with open(schema_path) as f:
        schema = json.loads(f.read())
    validate(instance=data, schema=schema)
    fallbacks = int(data.get("model_fallbacks", 0))
    anomalies = data.get("anomalies", []) or []
    if fallbacks > 0:
        print("[CANARY WARN] model_fallbacks:", fallbacks)
    if anomalies:
        print("[CANARY WARN] anomalies:", anomalies)
    print("[CANARY OK]")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(*sys.argv[1:3]))
    except ValidationError as e:
        print("[CANARY FAIL]", e.message)
        sys.exit(1)
