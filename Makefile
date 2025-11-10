#Makefile
SHELL := /bin/bash

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

fetch-fmu-models:
	if [ ! -d ./models/fmu-models ]; then \
		git clone git@code.ornl.gov:exadigit/fmu-models.git ./models/fmu-models; \
	else \
		git -C ./models/fmu-models pull; \
	fi
	
fetch-example-fmus:
	@echo "Fetching 'fmus' folder from POWER9CSM..."
	mkdir -p ./models/tmp
	curl -L -o ./models/tmp/POWER9CSM.zip https://code.ornl.gov/exadigit/POWER9CSM/-/archive/main/POWER9CSM-main.zip
	unzip -q ./models/tmp/POWER9CSM.zip -d ./models/tmp
	#rm -rf ./models/POWER9CSM
	mkdir -p ./models/POWER9CSM
	mv ./models/tmp/POWER9CSM-main/fmus ./models/POWER9CSM/fmus
	rm -rf ./models/tmp
	@echo "Copied 'fmus' folder from POWER9CSM → ./models/POWER9CSM"