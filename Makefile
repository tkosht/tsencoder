default: run

all: data run

data: convert

convert: up
	docker-compose exec app python convert.py

run: up
	docker-compose exec app python main.py

debug: up
	docker-compose exec app pudb3 main.py

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
	rm -f app/utils.py
	sudo rm -rf app/__pycache__
