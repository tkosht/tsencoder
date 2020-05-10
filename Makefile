default: up

all: run

# switch mode
gpu:
	@rm -f Dockerfile docker-compose.yml
	@ln -s docker/docker-compose.gpu.yml docker-compose.yml

cpu:
	@rm -f Dockerfile docker-compose.yml
	@ln -s docker/docker-compose.cpu.yml docker-compose.yml

# run tasks
run: up
	docker-compose exec app mlflow run --no-conda .

debug: up
	docker-compose exec app pudb3 encoder.py

# visualization tasks
ui: up
	docker-compose exec app mlflow ui --host=0.0.0.0

tb: up
	$(eval logdir:=$(shell ls -trd app/mlruns/0/* | tail -n 1 | perl -pe 's:^app/::'))
	echo $(logdir)
	docker-compose exec app tensorboard --host=0.0.0.0 --logdir=$(logdir)

# for docker-compose
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
