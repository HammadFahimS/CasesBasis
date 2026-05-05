.PHONY: setup start stop restart logs clean help

ENV_NAME = casesbasis

## Run this once to set up everything
setup:
	@echo "Creating conda environment (Python 3.11)..."
	conda env create -f environment.yml
	@echo "Starting FalkorDB..."
	docker compose up -d
	@echo ""
	@echo "Done. Activate your environment with:"
	@echo "  conda activate $(ENV_NAME)"
	@echo ""
	@echo "FalkorDB is running at localhost:6379"
	@echo "FalkorDB Browser UI at http://localhost:3000"

## Update conda env after changes to environment.yml
update:
	conda env update -f environment.yml --prune

## Start FalkorDB (Docker)
start:
	docker compose up -d

## Stop FalkorDB
stop:
	docker compose down

## Restart FalkorDB
restart:
	docker compose restart

## Tail FalkorDB logs
logs:
	docker compose logs -f falkordb

## Remove conda env and Docker volumes (destructive)
clean:
	conda env remove -n $(ENV_NAME)
	docker compose down -v

help:
	@grep -E '^##' Makefile | sed 's/## //'
