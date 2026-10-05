.PHONY: tune evaluate test all
tune:     ; python -m src.tune
evaluate: ; python -m src.evaluate
test:     ; python -m pytest -q tests
all: tune evaluate
