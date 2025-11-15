# Metacritic vs Steam Popularity Analysis 🎮

This project explores the relationship between **critic scores (Metacritic)** and **Steam player engagement**. Using real Steam and Metacritic data, it predicts which games are popular on Steam based on player activity, playtime, and reviews.  

The main finding is: **Steam popularity is mostly driven by player engagement—not critic scores.**

---

## 📂 Project Structure

MetacriticVSSteam/
├── analyze_games.py # Main Python script
├── steam_spy_data.csv # Steam game data
├── metacritic_Toppc_games.csv # Metacritic PC game data
├── graphs/ # Automatically generated graphs
└── README.md # Project documentation


---

## 🛠 Features & Functionality

1. **Data Loading** – Reads Steam and Metacritic datasets into Python using Pandas.  
2. **Title Matching** – Finds exact matches for games and uses fuzzy matching for slightly different names.  
3. **Popularity Definition** – Marks the top 25% of games by Steam owners as “popular”.  
4. **Feature Selection** – Uses metrics such as:
   - Positive reviews (`positive`)  
   - Negative reviews (`negative`)  
   - Average total playtime (`average_forever`)  
   - Median playtime (`median_forever`)  
   - Concurrent users (`ccu`)  
   - Critic scores (`score`)  

5. **Machine Learning** – Trains a **Random Forest Classifier** to predict whether a game is popular.  
6. **Dynamic Explanations** – Prints a detailed summary of model performance and feature importance in plain English.  
7. **Graph Export** – Generates readable graphs to show the most important factors and how engagement correlates with popularity.

---

## 🤖 How the Machine Learning Works

The goal is to **predict whether a game is “popular”** using features derived from player activity and reviews.

### 1. Preparing the Data
- We convert all relevant features into numbers.  
- Missing values are filled with the **median** of each column.  
- The target variable (`popular`) is **1** for top 25% of games by owners, **0** otherwise.

### 2. Splitting the Data
- The dataset is split into **training (75%)** and **testing (25%)** sets.  
- Training data teaches the model patterns; testing data evaluates its performance.

### 3. The Random Forest Classifier
- Random Forest is an ensemble method:
  - It trains **many decision trees** on random subsets of the data.  
  - Each tree predicts whether a game is popular.  
  - The final prediction is based on the **majority vote** of all trees.  

- Advantages for this project:
  - Handles non-linear relationships (e.g., playtime vs popularity).  
  - Automatically estimates feature importance.  
  - Robust to missing values and outliers.

### 4. Scaling Features
- Features are standardized so they all have similar scale.  
- This ensures the model treats each feature fairly, especially useful if features have very different ranges (e.g., playtime vs CCU).

### 5. Evaluating Performance
- **Metrics used:**
  - **Accuracy:** Overall correctness of predictions.
  - **Precision & Recall:** Measures for each class:
    - **Class 1 (Popular):** How well the model identifies truly popular games.  
    - **Class 0 (Not Popular):** How well the model identifies non-popular games.  
  - **AUC (Area Under the Curve):** Measures the model’s ability to rank popular games higher than non-popular ones.

- The script prints a **dynamic explanation**:
  - Whether the model is better at predicting popular or non-popular games.  
  - Which features had the strongest influence.  
  - Balanced or unbalanced recalls.

### 6. Understanding Feature Importance
The model automatically calculates how much each feature contributes to predictions:

| Feature            | Meaning |
|-------------------|---------|
| Positive reviews   | More positive reviews → more likely to be popular |
| Negative reviews   | Fewer negative reviews → higher popularity |
| CCU               | High concurrent users → active popularity |
| Average playtime  | Games people spend more hours on → popular |
| Median playtime   | Typical player engagement → consistency matters |
| Critic score      | Surprisingly, critic opinions have minimal effect |

---

## 📈 Graphs

The script generates and saves **graphs in the `graphs/` folder**:

1. **Feature Importance (%)** – Shows which factors most influence popularity.  
2. **Average Playtime vs Popularity** – Clear visualization of how player engagement correlates with being popular.

All graphs are saved automatically as PNG files when you run the script.

---