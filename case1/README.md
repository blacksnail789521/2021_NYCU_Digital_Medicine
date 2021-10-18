# Digital Medicine Case 1 - Obesity Status Detection
## Overview
- Two Tasks
    - Textual Task: Classify obesity as Present, Absent, Questionable, or Unmentioned based on explicitly documented information in the discharge summaries, e.g., the statement "the patient is obese".
    - Intuitive Task: Classify each disease as Present, Absent, or Questionable by applying their intuition and judgment to information in the discharge summaries, e.g., the statement "the patient weighs 230 lbs and is 5 ft 2 inches".
- Dataset
    -  Training data based on textual judgement (200 cases obesity vs. 200 cases unmentioned)
    -  Testing data based on intuitive judgement (200 cases obesity vs. 200 cases absence)
    -  Validation data (50 cases) based on textual judgement
-  Evaluation matrix
    -  F1-score
-  Method
    -  Rule-based
    -  Machine learning approaches
    -  Bert-based
    -  Majority vote

## Prepare
### Install Environment
```=bash
conda env create -f environment.yml
```

## Usage
### Bert
```=bash
python bert.py
python bert_doc_classification.py
python bert_model_list.py
```

### TF-IDF + Logistic regression
```=bash
python baseline.py
```

### Document classification using Bert
```=bash
python bert_doc_classification.py
```

### Majority vote
```=bash
python ensemble.py
```

### Rule-based approach
```=bash
python rule-based.py 
```
