# Makefile for PAN25 Docker Pipeline

IMAGE_NAME = pan25-app

# Default values (can be overridden on command line)
INPUT ?= $(PWD)/input
OUTPUT ?= $(PWD)/output

.PHONY: build run clean shell

# Build the Docker image
build:
	docker build -t $(IMAGE_NAME) .

# Run the pipeline with specified or default input/output
run:
	docker run --rm \
		-v $(abspath $(INPUT)):/input \
		-v $(abspath $(OUTPUT)):/output \
		$(IMAGE_NAME) -i /input -o /output

# Clean placeholder (no artifacts currently)
clean:
	echo "Nothing to clean."

# Interactive shell for debugging
shell:
	docker run --rm -it \
		-v $(abspath $(INPUT)):/input \
		-v $(abspath $(OUTPUT)):/output \
		$(IMAGE_NAME) /bin/bash
