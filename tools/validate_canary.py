import sys
import json
from jsonschema import validate, ValidationError


def main(meta_path: str = "reports/canary_run.meta.json", schema_path: str = "reports/canary.schema.json") -> int:
    data = json.loads(open(meta_path).read())
    schema = json.loads(open(schema_path).read())
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


