# Task 1 - System Ingestion & Model Environment Setup

## Objective
Set up a reproducible Machine Learning environment and perform a baseline experiment using the Iris dataset.

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Jupyter Notebook
- Git

## Project Structure

```
Task01_System_Ingestion/
│
├── data/
│   └── iris.csv
├── logs/
│   └── metrics.csv
├── models/
├── notebooks/
│   └── starter.ipynb
├── reports/
├── src/
│   └── train.py
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Workflow

1. Created a virtual environment.
2. Installed required Python packages.
3. Loaded the Iris dataset.
4. Verified dataset structure and class balance.
5. Split the dataset into Train, Validation, and Test sets.
6. Trained a DummyClassifier as a baseline model.
7. Evaluated the model using Accuracy.
8. Saved the evaluation metrics to `logs/metrics.csv`.

## Result

The baseline model completed successfully, and the evaluation metrics were saved in the `logs` folder.