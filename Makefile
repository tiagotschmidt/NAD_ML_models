# Define default target
all: init test clean

init:
	sudo chmod -R a+r /sys/class/powercap/intel-rapl

# Run tests
test:
	PYTHONPATH=. pytest -s

# Clean up
clean:
	rm *.log
	rm plot_results/* -rf
	rm models/json_models/*
	rm models/models_weights/*