# Clinically Informed Intermediate Reasoning (CIIR)
CIIR script creates an AUC value file and ROC curve image files from an intermediate feature file.

## Installation
```bash
pip install -r requirements.txt
```
## Usage
```bash
cd ./test
python AIgleason_sample.py
```

## Examples
- Inputs
  - Intermediate feature file: ./test/test_sample20.csv
  - Pretrained model file: ./test/bundle_sample.joblib
- Outputs
  - AUC value file: ./test/aucs.csv
  - ROC curve image files: 
    - ./test/roc_ci/roc_Combination_of_PSA_and_ML-predicted_reasoning-oriented_score.png
    - ./test/roc_ci/roc_ML-predicted_reasoning-oriented_score.png
    - ./test/roc_ci/roc_Tabular_data_of_100_variables_directly.png

## Licence
[Apache-2.0](https://choosealicense.com/licenses/apache-2.0/)
