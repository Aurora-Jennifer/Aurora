.PHONY: health test test-full lint pre-push

health:
	python scripts/repo_health.py

test:
	pytest -q

test-full:
	pytest -m "not quarantine" -q

lint:
	ruff check .

pre-push: health