.DEFAULT_GOAL := default
#################### PACKAGE ACTIONS ###################

SCRIPT_PATH = DeepSolarEye/handling/masks.py

train_test_val_split:
	python -c 'from DeepSolarEye.handling.split_data import train_test_val_split; train_test_val_split()'

.PHONY: run_process_mask
run_process_mask:
	@echo "Processing images and generating masks..."
	@python $(SCRIPT_PATH)
