# Makefile
.PHONY: help setup install test clean

help:
	@echo "Commands:"
	@echo "  init     : Enable the Intel RAPL Linux Directory."
	@echo "  install  : Installs the package in editable mode."
	@echo "  test     : Runs all tests with coverage."
	@echo "  clean    : Removes temporary files."

all: init test clean

init:
	sudo chmod -R a+r /sys/class/powercap/intel-rapl

setup:
	@echo "Setting up conda environment..."
	conda env create -f environment.yml

clean:
	@echo "Cleaning up..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

test:
	@echo "Running unit tests..."
	@pytest