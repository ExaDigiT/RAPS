#Makefile
.PHONY: pip run docker_build docker_run
all: pip

pip:
	pip install -e .

run:
	python3 ./main.py

test:
	pytest -n 8

IMAGE_NAME = raps

docker_build:
	docker build --platform linux/amd64 -t $(IMAGE_NAME) .

docker_run:
	docker run --platform linux/amd64 -it $(IMAGE_NAME)

