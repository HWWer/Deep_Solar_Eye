.DEFAULT_GOAL := default
#################### PACKAGE ACTIONS ###################

train_test_val_split:
	python -c 'from DeepSolarEye.handling.split_data import train_test_val_split; train_test_val_split()'

run_api:
	uvicorn DeepSolarEye.api.fast_api:app --reload
