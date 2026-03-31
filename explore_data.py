import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('data/cardio_train.csv', sep=';')
print("Cardiovascular Disease Dataset - Limitations & Bias Analysis")

print("\n1. Dataset Overview")
print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"\nColumn names and types:")
print(df.dtypes)

print("\n2. Missing Values Analysis")
missing = df.isnull().sum()
if missing.sum() == 0:
    print("No missing values detected")
else:
    print(missing[missing > 0])

print("\n3. Statistical Summary")
print(df.describe())

print("\n4. Data Quality Issues & Outliers")

# Age analysis (stored as days - convert to years)
df['age_years'] = df['age'] / 365.25
print(f"Age range: {df['age_years'].min():.1f} - {df['age_years'].max():.1f} years")

# Blood pressure outliers
print(f"\nSystolic BP (ap_hi):")
print(f"  Min: {df['ap_hi'].min()}, Max: {df['ap_hi'].max()}")
bp_extreme = df[(df['ap_hi'] > 250) | (df['ap_lo'] > 250)]
print(f"  Extreme values (>250): {len(bp_extreme)} records")
if len(bp_extreme) > 0:
    print("     Examples of extreme BP values:")
    print(bp_extreme[['id', 'ap_hi', 'ap_lo']].head())

# Height/Weight analysis
print(f"\nHeight range: {df['height'].min()} - {df['height'].max()} cm")
print(f"\nWeight range: {df['weight'].min()} - {df['weight'].max()} kg")
extreme_weight = len(df[df['weight'] < 30]) + len(df[df['weight'] > 150])
print(f"  Extreme weights (<30kg or >150kg): {extreme_weight} records")

# BMI analysis
df['BMI'] = df['weight'] / ((df['height'] / 100) ** 2)
print(f"\nBMI range: {df['BMI'].min():.1f} - {df['BMI'].max():.1f}")

print("\n5. Target Variable Distribution - CARDIO")
cardio_counts = df['cardio'].value_counts()
cardio_pct = df['cardio'].value_counts(normalize=True) * 100
print(f"Cardio Disease Present (1): {cardio_counts[1]} ({cardio_pct[1]:.1f}%)")
print(f"Cardio Disease Absent (0):  {cardio_counts[0]} ({cardio_pct[0]:.1f}%)")
if cardio_pct[1] > 60 or cardio_pct[1] < 40:
    print("  WARNING: Imbalanced classes may affect model training")

print("\n6. Gender Distribution")
gender_counts = df['gender'].value_counts().sort_index()
gender_pct = df['gender'].value_counts(normalize=True).sort_index() * 100
print(f"Gender 1 (likely Male):   {gender_counts.iloc[0]} ({gender_pct.iloc[0]:.1f}%)")
print(f"Gender 2 (likely Female): {gender_counts.iloc[1]} ({gender_pct.iloc[1]:.1f}%)")
if gender_pct.max() > 60:
    print("  WARNING: Significant gender imbalance - may bias model predictions")

print("\n7. Disease Distribution by Gender")
gender_disease = pd.crosstab(df['gender'], df['cardio'], normalize='index') * 100
print("Disease rate by gender:")
print(gender_disease)
print(f"  -> Gender 1 disease rate: {gender_disease.loc[1, 1]:.1f}%")
print(f"  -> Gender 2 disease rate: {gender_disease.loc[2, 1]:.1f}%")
if abs(gender_disease.loc[1, 1] - gender_disease.loc[2, 1]) > 10:
    print("  WARNING: Significant disease rate difference between genders")

print("\n8. Lifestyle Factors - Smoking, Alcohol, Physical Activity")
print(f"Smokers: {df['smoke'].sum()} ({df['smoke'].sum()/len(df)*100:.1f}%)")
print(f"Alcohol consumers: {df['alco'].sum()} ({df['alco'].sum()/len(df)*100:.1f}%)")
print(f"Physically active: {df['active'].sum()} ({df['active'].sum()/len(df)*100:.1f}%)")

print("\n9. Health Markers - Cholesterol & Glucose")
print(f"Cholesterol levels distribution:")
print(df['cholesterol'].value_counts().sort_index())
print(f"\nGlucose levels distribution:")
print(df['gluc'].value_counts().sort_index())
