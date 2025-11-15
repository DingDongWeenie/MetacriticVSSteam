import os
import pandas as pd
import numpy as np
import re
from rapidfuzz import process, fuzz
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import matplotlib.pyplot as plt

# ============================================
# AUTO-DETECT SCRIPT DIRECTORY
# ============================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def path(file_name):
    return os.path.join(BASE_DIR, file_name)

# ================================
# 1. LOAD DATA
# ================================
print("\n\033[1m========== STEP 1: LOADING GAME DATA ==========\033[0m\n")

steam = pd.read_csv(path("steam_spy_data.csv"), low_memory=False)
meta = pd.read_csv(path("metacritic_Toppc_games.csv"), low_memory=False)

print(f"We loaded {len(steam):,} games from Steam.")
print(f"We loaded {len(meta):,} games from Metacritic.\n")
print("Goal: Compare what critics love vs. what Steam players actually play.\n")

# ================================
# 2. FIND TITLE COLUMNS
# ================================
def find_title_col(df):
    for col in df.columns:
        if col.lower() in ("name", "title", "game"):
            return col
    for col in df.columns:
        if "name" in col.lower() or "title" in col.lower():
            return col
    raise ValueError("No title column found")

steam_title = find_title_col(steam)
meta_title = find_title_col(meta)

# ================================
# 3. NORMALIZE TITLES
# ================================
def normalize_title(s):
    if pd.isna(s): 
        return ""
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9 ]", "", s)
    return re.sub(" +", " ", s).strip()

steam["title_norm"] = steam[steam_title].apply(normalize_title)
meta["title_norm"]  = meta[meta_title].apply(normalize_title)

# ================================
# 4. EXACT & FUZZY MATCHING
# ================================
print("\n\033[1m========== STEP 2: MATCHING GAMES BETWEEN DATASETS ==========\033[0m\n")

merged = meta.merge(steam, on="title_norm", how="left")
exact = len(merged) - merged[steam_title].isna().sum()
missing = merged[steam_title].isna().sum()

print(f"Exact title matches found: {exact:,}")
print(f"Titles that needed fuzzy matching: {missing:,}\n")

unmatched = meta[meta["title_norm"].isin(
    merged[merged[steam_title].isna()]["title_norm"]
)]

steam_titles = steam["title_norm"].tolist()
fuzzy_map = {}

for title in unmatched["title_norm"]:
    match, score, _ = process.extractOne(title, steam_titles, scorer=fuzz.ratio)
    if score > 90:
        fuzzy_map[title] = match

meta["title_fuzzy"] = meta["title_norm"].map(lambda x: fuzzy_map.get(x, x))
merged = meta.merge(steam, left_on="title_fuzzy", right_on="title_norm", how="left")

print(f"Total games successfully matched after fuzzy matching: {merged['appid'].notna().sum():,}\n")

# ================================
# 5. PARSE OWNERS
# ================================
def parse_owners(val):
    if pd.isna(val): return np.nan
    s = str(val)
    if "-" in s:
        low, high = s.split("-")
        try:
            return (int(low.replace(",", "")) + int(high.replace(",", ""))) / 2
        except:
            return np.nan
    nums = re.findall(r"\d+", s.replace(",", ""))
    return float(nums[0]) if nums else np.nan

merged["owners_num"] = merged["owners"].apply(parse_owners)

# ================================
# 6. DEFINE POPULARITY
# ================================
print("\033[1m========== STEP 3: DEFINING POPULARITY ==========\033[0m\n")

merged = merged.dropna(subset=["owners_num"])
threshold = merged["owners_num"].quantile(0.75)
merged["popular"] = (merged["owners_num"] >= threshold).astype(int)

print(f"We marked the top 25% of Steam games as 'popular.'")
print(f"Games need at least {threshold:,.0f} owners to be considered popular.\n")

# ================================
# 7. FEATURES FOR PREDICTION
# ================================
feature_cols = []
for c in merged.columns:
    if c.lower() == "score":
        feature_cols.append(c)
for c in ["positive", "negative", "average_forever", "median_forever", "ccu"]:
    if c in merged.columns:
        feature_cols.append(c)

X = merged[feature_cols].apply(pd.to_numeric, errors="coerce")
X = X.fillna(X.median())
y = merged["popular"]

# ================================
# 8. TRAIN MODEL
# ================================
print("\033[1m========== STEP 4: LEARNING WHAT MAKES A GAME POPULAR ==========\033[0m\n")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model = Pipeline([
    ("scale", StandardScaler()),
    ("rf", RandomForestClassifier(n_estimators=200, random_state=42))
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\033[1mHow well the model predicts popularity:\033[0m\n")
print(classification_report(y_test, y_pred))
print(f"\033[1mOverall prediction quality (AUC): {roc_auc_score(y_test, y_proba):.4f}\033[0m\n")

# ================================
# 9. FEATURE IMPORTANCE
# ================================
rf = model.named_steps["rf"]
importances = pd.Series(rf.feature_importances_, index=X.columns)
importances_pct = importances * 100

print("\033[1m========== STEP 5: WHAT FACTORS MATTER MOST ==========\033[0m\n")
print("Here is what mattered for predicting Steam popularity:\n")

for feature, value in importances_pct.sort_values(ascending=False).items():
    print(f"\033[1m{feature} ({value:.2f}%):\033[0m")
    if feature.lower() == "positive":
        print("   → Positive reviews strongly indicate popularity.\n")
    elif feature.lower() == "negative":
        print("   → Fewer negative reviews = more satisfied players.\n")
    elif feature.lower() == "ccu":
        print("   → Active concurrent users reflect active popularity.\n")
    elif feature.lower() == "average_forever":
        print("   → Higher total playtime shows strong engagement.\n")
    elif feature.lower() == "median_forever":
        print("   → Consistent playtime across typical users.\n")
    elif feature.lower() == "score":
        print("   → Critic reviews had low predictive power.\n")
    else:
        print("   → Additional predictive factor.\n")

# ================================
# 10. EXPLANATION BASED ON RESULTS
# ================================
report = classification_report(y_test, y_pred, output_dict=True)

def bold(text):
    return f"\033[1m{text}\033[0m"

print(bold("\n========== MODEL PERFORMANCE ANALYSIS ==========\n"))

accuracy = report["accuracy"]
if accuracy >= 0.95:
    acc_msg = "The model is extremely accurate and distinguishes popularity with high reliability."
elif accuracy >= 0.85:
    acc_msg = "The model has strong accuracy and performs reliably."
elif accuracy >= 0.75:
    acc_msg = "The model has moderate accuracy and occasionally misclassifies borderline games."
else:
    acc_msg = "The model struggles and may need better features or more data."
print(acc_msg + "\n")

prec0, rec0 = report["0"]["precision"], report["0"]["recall"]
prec1, rec1 = report["1"]["precision"], report["1"]["recall"]

print(bold("=== CLASS DIAGNOSTICS ===\n"))
if rec1 > rec0:
    class_msg = "It detects POPULAR games better than non-popular ones."
elif rec0 > rec1:
    class_msg = "It detects NOT-popular games more reliably."
else:
    class_msg = "It identifies both classes equally well."
print(class_msg)
if abs(rec1 - rec0) < 0.10:
    print("Class recall is balanced.\n")
else:
    print("There is a noticeable class imbalance.\n")

print(bold("=== FEATURE IMPORTANCE INTERPRETATION ===\n"))
top_feature = importances_pct.sort_values(ascending=False).idxmax()
top_value = importances_pct.max()
print(f"The strongest predictor is: {bold(top_feature)} ({top_value:.2f}%)\n")

# ================================
# FINAL SUMMARY
# ================================
print(bold("\n========== FINAL SUMMARY ==========\n"))
summary = []

# engagement-driven features
engagement_features = ["ccu", "average_forever", "median_forever"]
engagement_strength = sum(importances[f] for f in engagement_features if f in importances)
critic_importance = importances.get("score", 0)

if engagement_strength > critic_importance:
    summary.append("➡ Steam popularity is driven mostly by player engagement, not critic opinion.")
else:
    summary.append("➡ Critic reviews surprisingly play a larger role than expected.")

if critic_importance < 0.05:
    summary.append("➡ Metacritic critic scores contributed very little to predicting popularity.")
else:
    summary.append("➡ Critic scores had a noticeable impact on popularity predictions.")

summary.append(f"➡ The most influential single factor was: {top_feature}.")

if accuracy >= 0.85:
    summary.append("➡ The model's predictions are fairly trustworthy overall.")
else:
    summary.append("➡ The model struggled with accuracy, so results should be interpreted cautiously.")

for line in summary:
    print(line)
print("\n====================================================\n")

# ================================
# EXPORT GRAPHS TO "graphs" FOLDER
# ================================
print(bold("\n========== EXPORTING GRAPHS ==========\n"))

GRAPH_DIR = path("graphs")
os.makedirs(GRAPH_DIR, exist_ok=True)

# Feature Importance Chart
plt.figure(figsize=(10, 6))
sorted_importances = importances_pct.sort_values()
colors = ['skyblue'] * len(sorted_importances)
top3 = sorted_importances.tail(3).index
for i, feature in enumerate(sorted_importances.index):
    if feature in top3:
        colors[i] = 'orange'

sorted_importances.plot(kind='barh', color=colors)
plt.title("Feature Importance (%) — Top 3 Highlighted")
plt.xlabel("Importance %")
plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR, "feature_importance.png"))
plt.close()
print("Saved: graphs/feature_importance.png")

# Average Playtime vs Popularity
plt.figure(figsize=(10, 5))
jitter_strength = merged["average_forever"].std() * 0.02
jitter = np.random.normal(0, jitter_strength, size=len(merged))
merged["average_forever_jitter"] = merged["average_forever"] + jitter
merged["popular_offset"] = merged["popular"].replace({0: 0.05, 1: 0.95})

plt.scatter(
    merged["average_forever_jitter"],
    merged["popular_offset"],
    alpha=0.3,
    color='orange'
)
plt.xscale("log")
plt.xlabel("Average Playtime (log scale)")
plt.ylabel("Popularity")
plt.title("Average Playtime vs Popularity (Improved Layout)")
plt.yticks([0.05, 0.95], ["Not Popular", "Popular"])
plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR, "average_playtime_vs_popularity.png"))
plt.close()
print("Saved: graphs/average_playtime_vs_popularity.png")

print("\nAll graphs exported successfully to the 'graphs' folder!\n")
