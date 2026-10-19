PYTHON ?= python

.PHONY: install test run docker

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest

run:
	$(PYTHON) -m uvicorn app.main:get_app --factory --host 0.0.0.0 --port 8000

docker:
	docker compose up --build
