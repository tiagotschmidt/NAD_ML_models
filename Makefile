# Define default target
all: init test clean

init:
	sudo chmod -R a+r /sys/class/powercap/intel-rapl

