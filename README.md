# Cardiovascular Disease Prognosis: ML-Based Decision Support System

## Overview

This project develops a **machine learning-based Decision Support System** that predicts a user's probability of having cardiovascular disease based on accessible health metrics. The system is designed to be **adaptive** (work with partial information and improves predictions as users provide more clinical data) and to output a **Risk Probability Percentage** rather than a rigid binary classification.

## Problem Statement

"Cardiovascular diseases are the leading cause of death globally. While early detection significantly improves survival rates, many individuals lack immediate access to complete clinical blood panels". 
This system bridges that gap by providing an adaptive risk assessment tool.

### Why Adaptive?
A typical user might know their height, weight, and age but not their exact glucose or cholesterol levels. Instead of failing, our system:
- Provides baseline risk probabilities with available inputs
- Updates predictions as more clinical data becomes available

## Dataset

- **Source**: [Cardiovascular Disease Dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset) (S. Ulianova on Kaggle)
- **Size**: 70,000 clinical records
- **Features**: 11 objective, subjective, and examination features including:
  - Demographics: age, gender
  - Anthropometric: height, weight
  - Clinical: blood pressure, cholesterol, glucose
  - Behavioral: smoking habits, physical activity

## Project Structure

```
├── main.ipynb                          # Central orchestration notebook
├── data/
│   ├── raw/
│   │   └── cardio_train.csv           # Original dataset
│   └── processed/
│       └── cardio_train_cleaned.csv   # Cleaned & preprocessed data
├── src/
│   ├── data/
│   │   └── explore.py                 # Data exploration & analysis
│   ├── features/
│   │   └── build_features.py          # Data cleaning & feature engineering
│   └── models/
│       ├── train.py                   # Model training functions
│       ├── predict.py                 # Prediction interface
│       ├── confidence_analysis.py     # Confidence scoring
│       └── threshold_analysis.py      # Threshold optimization
├── api/
│   └── main.py                        # REST API (in development)
├── frontend/
│   └── app.py                         # Web interface (in development)
├── docs/
│   └── Proposal.md                    # Full project proposal
└── models/                            # Trained model artifacts

```

## Team

- **[Duarte Branco - 119253](https://github.com/duartebranco)**
- **[Filipe Viseu - 119192](https://github.com/FilipeNV1)**
- **[Samuel Vinhas - 119405](https://github.com/samuelvinhas)**