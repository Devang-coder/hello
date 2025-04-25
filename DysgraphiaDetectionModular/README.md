# Dysgraphia Detection (Modular Version)

Modular implementation of a dysgraphia detection system using CNN.

## Folder Structure
- `dataset/dysgraphic/` - place dysgraphic images here
- `dataset/non_dysgraphic/` - place non-dysgraphic images here
- `src/` - source code files (main.py, model.py, etc.)
- `model/` - saved trained model will be stored here

## Setup & Run
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run training script:
```bash
python src/main.py
```