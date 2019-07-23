# PAWS
Protection Assistant for Wildlife Security
Prediction and prescription to combat illegal wildlife poaching.

Code by Lily Xu, Harvard University. June 2019

## Directories
- `./preprocess/` - R code for processing raw data
- `./preprocess_consolidate/` - Python code for consolidating output from preprocessing
- `./iware/` - Python code for ML risk prediction
- `./predict/` Python code for patrol planning

These directories must be made:
- `./inputs/` - raw data: CSV file of patrol observations and shapefiles of geographic features
- `./outputs/` - output of preprocessing step

## Processing order
In `./preprocess/`, execute the `pipeline` script to run all required preprocessing steps.

In `./consolidate_preprocess/`, execute the driver script, which will call all necessary functions.

In `./iware/`, execute the driver script, and choose whether to test (to run train/test code and evaluate performance) or make predictions. Run `visualize_maps.py` to generate images of the riskmaps.

In `./prediction/`, follow the README there.

In `./field_tests/`, execute `select_field_test.py` to run relevant tests
