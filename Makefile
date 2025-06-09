# Makefile
.PHONY: help init setup install update test clean

help:
	@echo "Commands:"
	@echo "  init     : Enable the Intel RAPL Linux Directory."
	@echo "  setup    : Creates the conda environment from scratch and installs the package."
	@echo "  install  : Installs the package in editable mode."
	@echo "  update   : Pulls latest changes and updates the conda environment."
	@echo "  test     : Runs all tests with coverage."
	@echo "  clean    : Removes temporary files."

all: init test clean

init:
	sudo chmod -R a+r /sys/class/powercap/intel-rapl

setup:
	@echo "Setting up conda environment..."
	conda env create -f environment.yml
	$(MAKE) install

install:
	@echo "Installing the package in editable mode..."
	pip install -e .

update:
	@echo "▶️ Pulling latest changes from the repository..."
	@git pull
	@echo "▶️ Updating conda environment from environment.yml..."
	@conda env update --file environment.yml --prune
	@echo "✅ Environment is up to date!"

clean:
	@echo "Cleaning up..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

test:
	@echo "Running unit tests..."
	@pytest