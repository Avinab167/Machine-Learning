import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# STEP 1: LOAD DATA
# ─────────────────────────────────────────────
print("=" * 55)
print("  LIFE EXPECTANCY PREDICTION — ML MODEL")
print("=" * 55)

df = pd.read_csv("C:/Users/User/OneDrive/Desktop/DV_group/Life Expectancy Data.csv")
df.columns = df.columns.str.strip()  # strip whitespace from column names
print(f"\n Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")

# ─────────────────────────────────────────────
# STEP 2: EXPLORE
# ─────────────────────────────────────────────
print("\n Target Variable (Life expectancy) stats:")
print(df['Life expectancy'].describe().round(2))

print(f"\n Missing values per column:")
missing = df.isnull().sum()
print(missing[missing > 0])

# ─────────────────────────────────────────────
# STEP 3: CLEAN DATA
# ─────────────────────────────────────────────
# Drop rows where target is missing
df = df.dropna(subset=['Life expectancy'])

# Fill remaining missing values with column median
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

print(f"\n After cleaning: {df.shape[0]} rows remaining")
print(f"   Missing values left: {df.isnull().sum().sum()}")

# ─────────────────────────────────────────────
# STEP 4: ENCODE CATEGORICAL FEATURES
# ─────────────────────────────────────────────
# Encode 'Status': Developing=0, Developed=1
le = LabelEncoder()
df['Status_encoded'] = le.fit_transform(df['Status'])

print(f"\n Encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# ─────────────────────────────────────────────
# STEP 5: SELECT FEATURES
# ─────────────────────────────────────────────
features = [
    'Adult Mortality',
    'infant deaths',
    'Alcohol',
    'percentage expenditure',
    'Hepatitis B',
    'BMI',
    'under-five deaths',
    'Polio',
    'Total expenditure',
    'Diphtheria',
    'HIV/AIDS',
    'GDP',
    'thinness  1-19 years',
    'thinness 5-9 years',
    'Income composition of resources',
    'Schooling',
    'Status_encoded'
]

X = df[features]
y = df['Life expectancy']

print(f"\n Features selected: {len(features)}")
print(f"   Target: 'Life expectancy'")

# ─────────────────────────────────────────────
# STEP 6: TRAIN/TEST SPLIT
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\n Train set: {X_train.shape[0]} samples")
print(f"   Test set:  {X_test.shape[0]} samples")

# ─────────────────────────────────────────────
# STEP 7: TRAIN MODELS
# ─────────────────────────────────────────────
print("\n" + "─" * 55)
print("  TRAINING MODELS...")
print("─" * 55)

# --- Model 1: Linear Regression ---
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

lr_r2   = r2_score(y_test, lr_pred)
lr_mae  = mean_absolute_error(y_test, lr_pred)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))

print(f"\nLinear Regression:")
print(f"   R² Score : {lr_r2:.4f}  ({lr_r2*100:.1f}% variance explained)")
print(f"   MAE      : {lr_mae:.2f} years")
print(f"   RMSE     : {lr_rmse:.2f} years")

# --- Model 2: Random Forest ---
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

rf_r2   = r2_score(y_test, rf_pred)
rf_mae  = mean_absolute_error(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))

print(f"\n Random Forest Regressor:")
print(f"   R² Score : {rf_r2:.4f}  ({rf_r2*100:.1f}% variance explained)")
print(f"   MAE      : {rf_mae:.2f} years")
print(f"   RMSE     : {rf_rmse:.2f} years")


# ─────────────────────────────────────────────
# STEP 9: SAMPLE PREDICTIONS
# ─────────────────────────────────────────────
print("\n" + "─" * 55)
print("  SAMPLE COUNTRY PREDICTIONS (Random Forest)")
print("─" * 55)

sample_countries = ['Nepal', 'United States of America', 'Japan', 'Afghanistan', 'Germany','Italy', 'Australia','Ethiopia']

for country in sample_countries:
    country_df = df[df['Country'] == country]
    if country_df.empty:
        print(f"\n    '{country}' not found in dataset")
        continue
    # Use most recent year's data
    row = country_df.sort_values('Year', ascending=False).iloc[0]
    X_sample = pd.DataFrame([row[features]])
    pred = rf.predict(X_sample)[0]
    actual = row['Life expectancy']
    year = int(row['Year'])
    print(f"\n   {country} ({year})")
    print(f"     Actual:    {actual:.1f} years")
    print(f"     Predicted: {pred:.1f} years")
    print(f"     Error:     {abs(pred - actual):.1f} years")

print("\n" + "=" * 55)
print("  DONE! Model training complete.")
print("=" * 55)
