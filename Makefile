default: up

all: run

run: up
	docker-compose exec app python enc.py

debug: up
	docker-compose exec app pudb3 enc.py

up:
	docker-compose up -d

active:
	docker-compose up

ps images down:
	docker-compose $@

im:images

build:
	docker-compose build --no-cache

reup: down up

clean:
	docker-compose down --rmi all
	sudo rm -rf app/__pycache__
