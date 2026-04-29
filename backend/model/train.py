import pandas as pd
import numpy as np
import pickle
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score

# PATH SETUP (IMPORTANT FIX)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

data_path_1 = os.path.join(BASE_DIR, "../../data/1.csv")
data_path_2 = os.path.join(BASE_DIR, "../../data/batch_22.csv")

# CLEANING FUNCTION
def clean_cols(df, cols):
    for c in cols:
        df[c] = df[c].astype(str).str.replace('*', '', regex=False)
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

# TREND FUNCTION
def compute_trend(row, cols):
    vals = row[cols].values.astype(float)
    x = np.arange(1, len(vals)+1)
    return np.polyfit(x, vals, 1)[0]

# LOAD DATA
df_full = pd.read_csv(data_path_1)   # S1–S8
df_s6 = pd.read_csv(data_path_2)     # S1–S6

# MODEL 1 → Predict S5, S6
print("\n==============================")
print("TRAINING MODEL 1 (S5, S6)")
print("==============================")

sgpa_early = ['S1_SGPA','S2_SGPA','S3_SGPA','S4_SGPA','S5_SGPA','S6_SGPA']

df_s6 = clean_cols(df_s6, sgpa_early + ['num_FF','num_AA'])
df_s6 = df_s6.dropna()

# Feature Engineering
df_s6['trend_4'] = df_s6.apply(lambda r: compute_trend(r, sgpa_early[:4]), axis=1)
df_s6['avg_4'] = df_s6[sgpa_early[:4]].mean(axis=1)
df_s6['max_4'] = df_s6[sgpa_early[:4]].max(axis=1)
df_s6['min_4'] = df_s6[sgpa_early[:4]].min(axis=1)

# Features & Target
X1 = df_s6[['S1_SGPA','S2_SGPA','S3_SGPA','S4_SGPA',
            'trend_4','avg_4','max_4','min_4',
            'num_FF','num_AA']]

y1 = df_s6[['S5_SGPA','S6_SGPA']]
names1 = df_s6['Name']

# Split
X1_train, X1_test, y1_train, y1_test, names1_train, names1_test = train_test_split(
    X1, y1, names1, test_size=0.2, random_state=42
)

# Train
model1 = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42)
model1.fit(X1_train, y1_train)

# Predict
pred1 = model1.predict(X1_test)

# Metrics
print("\nMODEL 1 PERFORMANCE")
print("MAE:", round(mean_absolute_error(y1_test, pred1), 3))
print("R2 :", round(r2_score(y1_test, pred1), 3))

# Results
results1 = pd.DataFrame({
    'Name': names1_test.values,
    'Actual_S5': y1_test['S5_SGPA'].values,
    'Pred_S5': pred1[:,0],
    'Actual_S6': y1_test['S6_SGPA'].values,
    'Pred_S6': pred1[:,1]
})

print("\n--- MODEL 1 RESULTS ---")
print(results1.round(2).to_string(index=False))


# MODEL 2 → Predict S7, S8, CGPA
print("\n==============================")
print("TRAINING MODEL 2 (S7, S8, CGPA)")
print("==============================")

sgpa_full = ['S1_SGPA','S2_SGPA','S3_SGPA','S4_SGPA',
             'S5_SGPA','S6_SGPA','S7_SGPA','S8_SGPA']

df_full = clean_cols(df_full, sgpa_full + ['CGPA','num_FF','num_AA'])
df_full = df_full.dropna()

# Feature Engineering
df_full['trend_6'] = df_full.apply(lambda r: compute_trend(r, sgpa_full[:6]), axis=1)
df_full['avg_6'] = df_full[sgpa_full[:6]].mean(axis=1)
df_full['max_6'] = df_full[sgpa_full[:6]].max(axis=1)
df_full['min_6'] = df_full[sgpa_full[:6]].min(axis=1)

# Features & Target
X2 = df_full[['S1_SGPA','S2_SGPA','S3_SGPA','S4_SGPA',
              'S5_SGPA','S6_SGPA',
              'trend_6','avg_6','max_6','min_6',
              'num_FF','num_AA']]

y2 = df_full[['S7_SGPA','S8_SGPA','CGPA']]
names2 = df_full['Name']

# Split
X2_train, X2_test, y2_train, y2_test, names2_train, names2_test = train_test_split(
    X2, y2, names2, test_size=0.2, random_state=42
)

# Train
model2 = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42)
model2.fit(X2_train, y2_train)

# Predict
pred2 = model2.predict(X2_test)

# Metrics
print("\nMODEL 2 PERFORMANCE")
print("MAE:", round(mean_absolute_error(y2_test, pred2), 3))
print("R2 :", round(r2_score(y2_test, pred2), 3))

# Results
results2 = pd.DataFrame({
    'Name': names2_test.values,
    'Actual_CGPA': y2_test['CGPA'].values,
    'Pred_CGPA': pred2[:,2]
})

print("\n--- MODEL 2 RESULTS ---")
print(results2.round(2).to_string(index=False))


# 💾 SAVE MODELS
model_dir = os.path.join(BASE_DIR, "../model")
os.makedirs(model_dir, exist_ok=True)

with open(os.path.join(model_dir, "model_stage1.pkl"), "wb") as f:
    pickle.dump(model1, f)

with open(os.path.join(model_dir, "model_stage2.pkl"), "wb") as f:
    pickle.dump(model2, f)

print("\n Models saved successfully!")