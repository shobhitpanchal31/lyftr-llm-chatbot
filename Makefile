.PHONY: up down logs build test

up:
	docker compose up --build -d

down:
	docker compose down

logs:
	docker compose logs -f

build:
	docker compose build

test:
	cd backend && pytest -v
