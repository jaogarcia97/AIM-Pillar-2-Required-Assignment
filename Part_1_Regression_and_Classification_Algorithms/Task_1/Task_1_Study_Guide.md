# 📚 Task 1 Study Guide: Linear Regression Fundamentals

> **A comprehensive recap of data analysis, statistics, and linear regression concepts**

---

## Table of Contents
1. [Loading Data with Pandas](#1-loading-data-with-pandas)
2. [Descriptive Statistics: Mean, Median, Standard Deviation](#2-descriptive-statistics)
3. [Distribution Plots & Vertical Lines (axvline)](#3-distribution-plots)
4. [Box Plots: How to Read Them](#4-box-plots)
5. [Correlation Matrix: The Complete Guide](#5-correlation-matrix)
6. [Feature Matrix (X) and Target Vector (y)](#6-feature-matrix-and-target-vector)
7. [What is "Training" a Model?](#7-what-is-training-a-model)
8. [Linear Regression Coefficients](#8-linear-regression-coefficients)
9. [Model Evaluation Metrics: MSE, RMSE, MAE, R²](#9-model-evaluation-metrics)
10. [Train vs Test Evaluation](#10-train-vs-test-evaluation)
11. [OLS Regression Results: How to Interpret](#11-ols-regression-results)
12. [Residuals: What They Are and Why They Matter](#12-residuals)
13. [Key Code Functions Cheat Sheet](#13-key-code-functions)
14. [Quick Reference Summary](#14-quick-reference-summary)

---

## 1. Loading Data with Pandas

### What is Pandas?
Pandas is a Python library for data manipulation. Think of it as **Excel in Python**.

### Loading an Excel File
```python
import pandas as pd

# Load Excel file into a DataFrame (like a spreadsheet)
df = pd.read_excel('housing_prices.xlsx')

# For CSV files:
df = pd.read_csv('data.csv')
```

### Common First Steps After Loading
```python
# See the first 5 rows
df.head()

# See the shape (rows, columns)
df.shape  # Returns (500, 5) means 500 rows, 5 columns

# See column names
df.columns.tolist()

# See data types and null counts
df.info()

# See statistical summary
df.describe()
```

### What `df.describe()` Shows You:
| Statistic | Meaning |
|-----------|---------|
| **count** | Number of non-null values |
| **mean** | Average value |
| **std** | Standard deviation (spread of data) |
| **min** | Minimum value |
| **25%** | 25th percentile (Q1) |
| **50%** | 50th percentile (median) |
| **75%** | 75th percentile (Q3) |
| **max** | Maximum value |

---

## 2. Descriptive Statistics

### Mean (Average)
**What it is:** Sum of all values divided by the count.

**Formula:** `mean = Σx / n`

**Code:**
```python
mean_price = df['House_Price'].mean()
```

**Interpretation:** "The average house price is $984,161"

**When to use:** When data is roughly symmetric (no extreme outliers).

---

### Median (Middle Value)
**What it is:** The middle value when data is sorted. 50% of values are below, 50% above.

**Code:**
```python
median_price = df['House_Price'].median()
```

**Interpretation:** "Half the houses cost less than $972,915, half cost more"

**When to use:** When data has outliers or is skewed (median is more robust).

---

### Mean vs Median: Key Insight! 🔑

| Scenario | Relationship | What it tells you |
|----------|--------------|-------------------|
| Mean ≈ Median | Data is symmetric | Normal distribution |
| Mean > Median | Right-skewed | Some very high values pulling mean up |
| Mean < Median | Left-skewed | Some very low values pulling mean down |

**Example:** If mean house price = $984,161 and median = $972,915:
- Mean > Median → Some expensive houses are pulling the average up
- The distribution is **slightly right-skewed**

---

### Standard Deviation (std)
**What it is:** Measures how spread out the data is from the mean.

**Think of it as:** "On average, how far are values from the average?"

**Code:**
```python
std_price = df['House_Price'].std()
```

**Interpretation:**
- **Low std:** Data points are clustered close to the mean (consistent)
- **High std:** Data points are spread out (high variability)

**Example:** std = $424,560 means:
- House prices typically vary by about $424,560 from the average
- Most houses (68% in a normal distribution) fall within ±1 std of the mean
- That's roughly $559,600 to $1,408,722

---

### Variance
**What it is:** Standard deviation squared (std²).

**Formula:** `variance = std²`

**Code:**
```python
variance = df['House_Price'].var()
```

**Why it matters:** 
- Variance is used in many statistical calculations
- **R² (R-squared)** tells you what percentage of variance in house prices your model explains

**Example:** If variance in House_Price = $180,251,702,784 and your model has R² = 0.85:
- Your model explains 85% of why house prices differ
- The remaining 15% is unexplained variance (random noise, missing features)

---

## 3. Distribution Plots

### What is a Distribution?
A distribution shows **how values are spread across a range**. It answers: "What values are common? What values are rare?"

### Histogram
```python
plt.hist(df['House_Price'], bins=30)
plt.xlabel('House Price')
plt.ylabel('Frequency')
plt.title('Distribution of House Prices')
plt.show()
```

**How to read:**
- **X-axis:** The values (house prices)
- **Y-axis:** How often each value range occurs
- **Tall bars:** Common values
- **Short bars:** Rare values

### KDE Plot (Kernel Density Estimate)
```python
import seaborn as sns
sns.kdeplot(df['House_Price'])
```

**What it is:** A smooth curve showing the distribution (like a smoothed histogram).

### The `axvline` Function

**What it does:** Draws a **vertical line** on your plot at a specific x-value.

```python
# Draw vertical line at the mean
plt.axvline(x=mean_price, color='red', linestyle='--', label='Mean')

# Draw vertical line at the median
plt.axvline(x=median_price, color='green', linestyle='-', label='Median')

plt.legend()
```

**Why use it:**
- Visually show where mean and median fall in the distribution
- Compare different statistics on the same plot
- Highlight important thresholds

**Visual Result:**
```
        │
    ▄▄▄▄│▄▄▄▄
   ▀████│███▀
        │
        │← Mean (red dashed line)
        │
```

---

## 4. Box Plots: How to Interpret Them

### What is a Box Plot?
A box plot (box-and-whisker plot) shows the **distribution of data in 5 key numbers**.

```python
# Create a box plot
plt.boxplot(df['House_Price'])

# Or with seaborn (prettier)
sns.boxplot(y=df['House_Price'])
```

### Anatomy of a Box Plot

```
                    ●  ← Outlier (above 1.5×IQR from Q3)
                    |
          ┌─────────┬─────────┐  ← Maximum whisker (Q3 + 1.5×IQR)
          │         │         │
          │    Q3 ──┼──       │  ← 75th percentile (top of box)
          │         │         │
          │  Median ┼─        │  ← 50th percentile (line inside box)
          │         │         │
          │    Q1 ──┼──       │  ← 25th percentile (bottom of box)
          │         │         │
          └─────────┴─────────┘  ← Minimum whisker (Q1 - 1.5×IQR)
                    |
                    ●  ← Outlier (below 1.5×IQR from Q1)
```

### The 5 Key Numbers

| Component | What it represents |
|-----------|-------------------|
| **Minimum whisker** | Smallest value within 1.5×IQR of Q1 |
| **Q1 (25th percentile)** | 25% of data is below this |
| **Median (Q2, 50th percentile)** | Middle value - 50% below, 50% above |
| **Q3 (75th percentile)** | 75% of data is below this |
| **Maximum whisker** | Largest value within 1.5×IQR of Q3 |

### IQR (Interquartile Range)
```
IQR = Q3 - Q1
```
**What it is:** The range containing the middle 50% of your data.

### Outliers (the dots)
Points plotted **outside the whiskers** are potential outliers.
- **Above:** value > Q3 + 1.5×IQR
- **Below:** value < Q1 - 1.5×IQR

### How to Interpret a Box Plot

**Example: House Prices Box Plot**

1. **Position of median line:**
   - If median is in the middle of the box → symmetric distribution
   - If median is closer to Q1 → right-skewed (some high values)
   - If median is closer to Q3 → left-skewed (some low values)

2. **Box size (IQR):**
   - Wide box = high variability in the middle 50%
   - Narrow box = consistent values in the middle 50%

3. **Whisker length:**
   - Equal whiskers = symmetric tails
   - Unequal whiskers = skewed distribution

4. **Outliers:**
   - Dots above = unusually expensive houses
   - Dots below = unusually cheap houses

---

## 5. Correlation Matrix: The Complete Guide

### What is Correlation?
Correlation measures **how two variables move together**.

**Correlation coefficient (r)** ranges from **-1 to +1**.

| Value | Meaning |
|-------|---------|
| **+1** | Perfect positive correlation (when X goes up, Y goes up) |
| **+0.7 to +1** | Strong positive correlation |
| **+0.4 to +0.7** | Moderate positive correlation |
| **+0.1 to +0.4** | Weak positive correlation |
| **0** | No correlation |
| **-0.1 to -0.4** | Weak negative correlation |
| **-0.4 to -0.7** | Moderate negative correlation |
| **-0.7 to -1** | Strong negative correlation |
| **-1** | Perfect negative correlation (when X goes up, Y goes down) |

### Creating a Correlation Matrix

```python
# Calculate correlation matrix
correlation_matrix = df.corr()
print(correlation_matrix)
```

**Output looks like:**
```
              Square_Feet  Num_Bedrooms  Num_Bathrooms  Age_of_House  House_Price
Square_Feet         1.00          0.05           0.02         -0.01         0.87
Num_Bedrooms        0.05          1.00           0.00          0.00         0.12
Num_Bathrooms       0.02          0.00           1.00          0.01         0.10
Age_of_House       -0.01          0.00           0.01          1.00        -0.20
House_Price         0.87          0.12           0.10         -0.20         1.00
```

### Reading the Correlation Matrix

#### The Diagonal (always 1.00)
- **Why?** Every variable is perfectly correlated with itself
- Square_Feet vs Square_Feet = 1.00 (obviously!)
- **Ignore the diagonal** when analyzing relationships

#### Off-Diagonal Values
These are the important ones! They show relationships between DIFFERENT variables.

**Example Reading:**
- Square_Feet & House_Price = **0.87** → Strong positive correlation
  - "Bigger houses cost more" ✓
- Age_of_House & House_Price = **-0.20** → Weak negative correlation
  - "Older houses cost slightly less" ✓

### Heatmap Visualization

```python
# Create a heatmap (visual correlation matrix)
import seaborn as sns

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, 
            annot=True,      # Show numbers
            cmap='coolwarm', # Color scheme
            center=0,        # Center color at 0
            vmin=-1, vmax=1) # Set color range
plt.title('Correlation Matrix Heatmap')
plt.show()
```

### Understanding the Color Map (cmap)

**'coolwarm' color scheme:**
- **Dark Blue:** Strong negative correlation (-1)
- **Light Blue:** Weak negative correlation
- **White:** No correlation (0)
- **Light Red:** Weak positive correlation
- **Dark Red:** Strong positive correlation (+1)

**Why center=0?**
So that white/neutral color represents zero correlation.

### What Correlation is FOR

1. **Feature Selection:** Find which features most affect your target
   - High |correlation| with House_Price → useful predictor

2. **Multicollinearity Detection:** Find features that correlate with each other
   - If two features have r > 0.8, they might be redundant

3. **Understanding Relationships:** Answer questions like:
   - "Does house size affect price?" (yes, r=0.87)
   - "Do older houses cost less?" (slightly, r=-0.20)

### The `.corr()` Function

```python
# Correlation with everything
df.corr()

# Correlation of one column with all others
df.corrwith(df['House_Price'])

# Specific correlation between two columns
df['Square_Feet'].corr(df['House_Price'])  # Returns 0.87
```

---

## 6. Feature Matrix (X) and Target Vector (y)

### The Big Picture

In machine learning, we split our data into:
- **Features (X):** The inputs/predictors we use to make predictions
- **Target (y):** The output we're trying to predict

### Visual Representation

```
Original Dataset:
┌─────────────┬─────────────┬──────────────┬─────────────┬─────────────┐
│ Square_Feet │ Num_Bedrooms│ Num_Bathrooms│ Age_of_House│ House_Price │
├─────────────┼─────────────┼──────────────┼─────────────┼─────────────┤
│    1660     │      1      │      2       │     38      │  769,899    │
│    4572     │      3      │      2       │     14      │ 1,598,878   │
│    3892     │      4      │      3       │     28      │ 1,326,075   │
└─────────────┴─────────────┴──────────────┴─────────────┴─────────────┘
         ↓                                                    ↓
        Features (X)                                      Target (y)
   (what we use to predict)                           (what we predict)
```

### Feature Matrix (X)

**What it is:** A 2D array (matrix) containing all the input features.

```python
# Select feature columns
X = df[['Square_Feet', 'Num_Bedrooms', 'Num_Bathrooms', 'Age_of_House']]

# Shape: (500, 4) = 500 samples, 4 features
print(X.shape)
```

**Why it's called a "matrix":**
- It has multiple rows (samples/houses)
- It has multiple columns (features/characteristics)
- It's 2-dimensional

### Target Vector (y)

**What it is:** A 1D array (vector) containing the values we want to predict.

```python
# Select target column
y = df['House_Price']

# Shape: (500,) = 500 values
print(y.shape)
```

**Why it's called a "vector":**
- It has only one column
- It's 1-dimensional (just a list of values)

### Simple vs Multiple Regression

**Simple Linear Regression:** One feature predicting target
```python
X = df[['Square_Feet']]  # Note: still needs double brackets!
y = df['House_Price']
```

**Multiple Linear Regression:** Multiple features predicting target
```python
X = df[['Square_Feet', 'Num_Bedrooms', 'Num_Bathrooms', 'Age_of_House']]
y = df['House_Price']
```

---

## 7. What is "Training" a Model?

### The Concept

**Training = Teaching the model to find patterns in your data**

Think of it like this:
1. You show the model many examples (X) with correct answers (y)
2. The model learns the relationship between X and y
3. Now it can predict y for new, unseen X values

### What Actually Happens During Training?

For Linear Regression, training means **finding the best line (or plane) that fits your data**.

**The model finds:**
- **Intercept (b₀):** Where the line crosses the y-axis
- **Coefficients (b₁, b₂, ...):** The slope for each feature

**The equation being learned:**
```
House_Price = b₀ + b₁(Square_Feet) + b₂(Num_Bedrooms) + b₃(Num_Bathrooms) + b₄(Age_of_House)
```

### The Training Process

```python
from sklearn.linear_model import LinearRegression

# Step 1: Create the model (empty, knows nothing)
model = LinearRegression()

# Step 2: Train the model (learn from data)
model.fit(X_train, y_train)  # This is where the "learning" happens!

# After training, the model knows:
print(model.intercept_)     # b₀
print(model.coef_)          # [b₁, b₂, b₃, b₄]
```

### What `.fit()` Does Internally

1. **Looks at all training data**
2. **Tries different coefficient values**
3. **Finds the combination that minimizes errors** (usually using a method called "Ordinary Least Squares")
4. **Stores the optimal coefficients** in the model

### Why Split into Train and Test?

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,      # 20% for testing
    random_state=42     # For reproducibility
)
```

**Purpose:**
| Set | What it's for |
|-----|--------------|
| **Training set (80%)** | Model learns from this data |
| **Test set (20%)** | We evaluate the model on this UNSEEN data |

**Why?** To see if the model generalizes to new data, not just memorizes training data.

---

## 8. Linear Regression Coefficients

### What Are Coefficients?

**Coefficients tell you HOW MUCH the target changes when a feature changes by 1 unit.**

**The model equation:**
```
House_Price = -126,142 + 334(Square_Feet) + 2,341(Bedrooms) + 16,789(Bathrooms) - 4,123(Age)
```

### Reading Each Coefficient

```python
# Get coefficients
print(f"Intercept: {model.intercept_}")
print(f"Coefficients: {model.coef_}")

# Better view:
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef:.2f}")
```

### Interpreting the Square_Feet Coefficient

**If Square_Feet coefficient = 334:**

**Meaning:** For every additional square foot, house price increases by **$334** (holding all other features constant).

**Real-world interpretation:**
- A house with 2000 sq ft vs 2001 sq ft differs by ~$334
- A house with 2000 sq ft vs 2100 sq ft differs by ~$33,400

### Why a Positive Coefficient Matters

**Question 1 from Task 1:** "If the coefficient for Square_Feet is significantly positive, what could be the reason?"

**Answer:**
1. **Supply and demand:** Larger homes are more desirable and scarce
2. **Cost to build:** More materials, more labor, more land
3. **More living space:** More rooms, more utility
4. **Market psychology:** Bigger = better perception

### Negative Coefficients

**If Age_of_House coefficient = -4,123:**

**Meaning:** For every additional year of age, house price decreases by **$4,123**.

**Why negative?**
- Older houses need more repairs
- Outdated designs/systems
- Higher maintenance costs

### Coefficient Significance

**What does "significant" mean?**
A coefficient is statistically significant if it's unlikely to be zero by chance.

**p-value < 0.05** → Coefficient is significant
- "We're 95% confident this feature actually affects the price"

**p-value > 0.05** → Coefficient might be zero (not significant)
- "This feature might not actually affect the price"

---

## 9. Model Evaluation Metrics

### Why Evaluate Models?

After training, we need to know: **How good are the predictions?**

### MSE (Mean Squared Error)

**What it is:** Average of squared differences between predicted and actual values.

**Formula:**
```
MSE = (1/n) × Σ(actual - predicted)²
```

**Code:**
```python
from sklearn.metrics import mean_squared_error

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
```

**Interpretation:**
- **Lower = Better** (less error)
- Units are squared (hard to interpret directly)
- Penalizes large errors more than small errors

**Example:** MSE = 18,924,305,284
- Hard to interpret because it's in "dollars squared"

---

### RMSE (Root Mean Squared Error)

**What it is:** Square root of MSE - brings units back to original scale.

**Formula:**
```
RMSE = √MSE
```

**Code:**
```python
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# Or
rmse = mean_squared_error(y_test, y_pred, squared=False)
```

**Interpretation:**
- **Lower = Better**
- Units are same as target (dollars)
- "On average, predictions are off by $X"

**Example:** RMSE = $137,564
- "On average, the model's predictions are off by about $137,564"

---

### MAE (Mean Absolute Error)

**What it is:** Average of absolute differences between predicted and actual.

**Formula:**
```
MAE = (1/n) × Σ|actual - predicted|
```

**Code:**
```python
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, y_pred)
```

**Interpretation:**
- **Lower = Better**
- Units are same as target (dollars)
- Less sensitive to outliers than MSE/RMSE

**Example:** MAE = $108,234
- "On average, predictions are off by about $108,234"

---

### R² (R-Squared) - Coefficient of Determination

**What it is:** Proportion of variance in the target explained by the model.

**Range:** 0 to 1 (can be negative for very bad models)

**Formula:**
```
R² = 1 - (SS_residual / SS_total)
```

**Code:**
```python
from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)
```

**Interpretation:**
| R² Value | Meaning |
|----------|---------|
| **1.0** | Perfect predictions (explains 100% of variance) |
| **0.9** | Excellent model (explains 90% of variance) |
| **0.7** | Good model (explains 70% of variance) |
| **0.5** | Moderate model (explains 50% of variance) |
| **0.3** | Weak model (explains 30% of variance) |
| **0.0** | Model is no better than predicting the mean |
| **< 0** | Model is worse than predicting the mean! |

**Example:** R² = 0.85
- "The model explains 85% of the variation in house prices"
- "15% of variation is due to factors not in our model"

---

### Comparison Summary

| Metric | Best Value | Units | Use When |
|--------|------------|-------|----------|
| **MSE** | 0 | Squared units | Penalize large errors more |
| **RMSE** | 0 | Original units | Interpret error in original scale |
| **MAE** | 0 | Original units | Robust to outliers |
| **R²** | 1 | Percentage | Explain model's explanatory power |

---

## 10. Train vs Test Evaluation

### Why Evaluate on Both?

| Evaluation | What it tells you |
|------------|-------------------|
| **Training score** | How well model learned the training data |
| **Testing score** | How well model generalizes to new data |

### Comparing Train and Test Performance

```python
# Training performance
y_train_pred = model.predict(X_train)
train_r2 = r2_score(y_train, y_train_pred)

# Testing performance
y_test_pred = model.predict(X_test)
test_r2 = r2_score(y_test, y_test_pred)

print(f"Training R²: {train_r2:.4f}")
print(f"Testing R²: {test_r2:.4f}")
```

### Interpreting the Comparison

| Scenario | Diagnosis | What to do |
|----------|-----------|------------|
| Train ≈ Test (both good) | ✅ Good fit | Model is working well |
| Train >> Test | ⚠️ Overfitting | Model memorized training data |
| Train ≈ Test (both bad) | ⚠️ Underfitting | Model is too simple |
| Train < Test | 🤔 Unusual | Check for data leakage |

**Example:**
- Training R² = 0.89
- Testing R² = 0.85

**Interpretation:** Small gap is normal. Model generalizes well.

---

## 11. OLS Regression Results: How to Interpret

### What is OLS?

**OLS = Ordinary Least Squares**

It's the method used to find the best-fitting line by minimizing the sum of squared residuals.

### Running OLS in Python

```python
import statsmodels.api as sm

# Add constant (intercept) to features
X_with_const = sm.add_constant(X)

# Fit OLS model
ols_model = sm.OLS(y, X_with_const).fit()

# Print summary
print(ols_model.summary())
```

### Reading the OLS Summary

```
                            OLS Regression Results                            
==============================================================================
Dep. Variable:            House_Price   R-squared:                       0.855
Model:                            OLS   Adj. R-squared:                  0.854
Method:                 Least Squares   F-statistic:                     731.1
Date:                Thu, 08 Jan 2026   Prob (F-statistic):          2.15e-198
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const      -1.261e+05   3.38e+04     -3.735      0.000   -1.92e+05   -5.97e+04
Square_Feet   334.12     10.234     32.651      0.000     313.98     354.26
Num_Bedrooms  2341.15   1204.56      1.943      0.053    -28.12    4710.42
Num_Bathrooms 16789.23  4521.34      3.714      0.000    7916.45   25662.01
Age_of_House -4123.56    892.13     -4.622      0.000   -5877.34   -2369.78
==============================================================================
```

### Key Statistics Explained

#### Top Section

| Statistic | Meaning | Example |
|-----------|---------|---------|
| **Dep. Variable** | Target variable | House_Price |
| **R-squared** | Variance explained | 0.855 (85.5%) |
| **Adj. R-squared** | R² adjusted for # of features | 0.854 |
| **F-statistic** | Overall model significance test | 731.1 |
| **Prob (F-statistic)** | p-value for F-test | 2.15e-198 (very small = significant) |

#### Coefficient Table

| Column | Meaning |
|--------|---------|
| **coef** | The coefficient value |
| **std err** | Standard error (uncertainty of estimate) |
| **t** | t-statistic (coef / std err) |
| **P>\|t\|** | p-value - is coefficient significant? |
| **[0.025 0.975]** | 95% confidence interval |

### Reading P-values (P>|t|)

**The magic number: 0.05**

| P-value | Interpretation |
|---------|----------------|
| **< 0.001** | *** Highly significant |
| **< 0.01** | ** Very significant |
| **< 0.05** | * Significant |
| **≥ 0.05** | Not statistically significant |

**Example from above:**
- Square_Feet: P = 0.000 → **Highly significant** (definitely affects price)
- Num_Bedrooms: P = 0.053 → **Not quite significant** (might not affect price)

### What "0.05" Means

**P-value = 0.05 means:**
- There's a 5% chance the coefficient is actually zero
- We're 95% confident the feature has a real effect

---

## 12. Residuals: What They Are and Why They Matter

### What is a Residual?

**Residual = Actual Value - Predicted Value**

```
residual = y_actual - y_predicted
```

**Think of it as:** The "error" or "leftover" that the model couldn't explain.

### Calculating Residuals

```python
# Get predictions
y_pred = model.predict(X)

# Calculate residuals
residuals = y - y_pred

# Or get from OLS model
residuals = ols_model.resid
```

### Residual Plot

```python
# Residual plot
plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()
```

### How to Read a Residual Plot

**What you WANT to see (good model):**
```
      |     •   •        
      |  •    •    •  •   
  0 ──┼─────•──•────•─────  ← Residuals randomly scattered around 0
      |  •     •   •     •
      |    •      •       
      └────────────────────
             Predicted
```

**Signs of problems:**

1. **Funnel shape (heteroscedasticity):**
```
      |           • •  • 
      |        •    •    •  
  0 ──┼──•──•───────────────
      |   •  •           
      |  •               
```
- Variance increases with predicted value
- Model is less reliable for high predictions

2. **Curved pattern (non-linearity):**
```
      |  •  •        • •  
      |     •  •    •    
  0 ──┼────────•──•────────
      |        •  •      
      |  •  •        • • 
```
- Linear model doesn't capture the true relationship
- Need polynomial terms or different model

3. **Clusters:**
- Suggests missing categorical variable

### Residual Statistics (from OLS)

```
Residual Statistics:
  Min       -412534.23
  1Q        -89123.45
  Median     2341.23
  3Q         91234.56
  Max        523412.78
```

**Interpretation:**
- **Min/Max:** Range of errors
- **Median near 0:** Good sign (errors are balanced)
- **1Q/3Q:** Middle 50% of errors

---

## 13. Key Code Functions Cheat Sheet

### Data Loading

```python
# Read Excel file
df = pd.read_excel('file.xlsx')

# Read CSV file
df = pd.read_csv('file.csv')
```

### Data Exploration

```python
# First 5 rows
df.head()

# Last 5 rows
df.tail()

# Shape (rows, columns)
df.shape

# Column info
df.info()

# Statistics
df.describe()

# Check for nulls
df.isnull().sum()
```

### Statistical Functions

```python
# Mean (average)
df['column'].mean()

# Median (middle value)
df['column'].median()

# Standard deviation
df['column'].std()

# Variance
df['column'].var()

# Correlation matrix
df.corr()

# Specific correlation
df['col1'].corr(df['col2'])
```

### Visualization

```python
# Histogram
plt.hist(df['column'], bins=30)

# Vertical line
plt.axvline(x=value, color='red', linestyle='--')

# Box plot
plt.boxplot(df['column'])
# or
sns.boxplot(y=df['column'])

# Correlation heatmap
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')

# Scatter plot
plt.scatter(x, y)
```

### Machine Learning

```python
# Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create and train model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

# Get coefficients
model.intercept_  # b₀
model.coef_       # [b₁, b₂, ...]

# Make predictions
y_pred = model.predict(X_test)
```

### Evaluation Metrics

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# MSE
mse = mean_squared_error(y_true, y_pred)

# RMSE
rmse = np.sqrt(mse)

# MAE
mae = mean_absolute_error(y_true, y_pred)

# R²
r2 = r2_score(y_true, y_pred)
```

### numpy.polyfit

```python
# Fit polynomial
# polyfit(x, y, degree)
coefficients = np.polyfit(x, y, 1)  # Linear (degree 1)
# Returns [slope, intercept]

coefficients = np.polyfit(x, y, 2)  # Quadratic (degree 2)
# Returns [a, b, c] for ax² + bx + c
```

### OLS Regression

```python
import statsmodels.api as sm

# Add constant for intercept
X_const = sm.add_constant(X)

# Fit model
ols_model = sm.OLS(y, X_const).fit()

# Print full summary
print(ols_model.summary())

# Get specific values
ols_model.params      # Coefficients
ols_model.pvalues     # P-values
ols_model.rsquared    # R²
ols_model.resid       # Residuals
```

---

## 14. Quick Reference Summary

### The Complete Data Science Process

```
1. LOAD DATA
   └─ pd.read_excel() / pd.read_csv()

2. EXPLORE DATA
   ├─ df.head(), df.info(), df.describe()
   ├─ Check for missing values
   └─ Understand distributions (histograms, box plots)

3. ANALYZE RELATIONSHIPS
   ├─ Correlation matrix (df.corr())
   ├─ Heatmap visualization
   └─ Identify important features

4. PREPARE DATA
   ├─ Define X (features) and y (target)
   └─ Split into train/test sets

5. BUILD MODEL
   ├─ Create model: LinearRegression()
   └─ Train model: model.fit(X_train, y_train)

6. MAKE PREDICTIONS
   └─ y_pred = model.predict(X_test)

7. EVALUATE MODEL
   ├─ Calculate metrics (MSE, RMSE, MAE, R²)
   ├─ Compare train vs test performance
   └─ Analyze coefficients

8. CHECK ASSUMPTIONS
   ├─ Plot residuals
   └─ Look for patterns (non-linearity, heteroscedasticity)

9. INTERPRET & REPORT
   ├─ OLS summary for detailed statistics
   └─ Answer the business questions
```

### One-Line Summaries

| Concept | One-Line Summary |
|---------|------------------|
| **Mean** | Average value |
| **Median** | Middle value (50th percentile) |
| **Standard Deviation** | How spread out values are from the mean |
| **Variance** | Standard deviation squared |
| **Correlation** | How two variables move together (-1 to +1) |
| **Feature Matrix (X)** | Input data used to make predictions |
| **Target Vector (y)** | Output data we want to predict |
| **Training** | Teaching the model to find patterns |
| **Coefficient** | How much target changes per unit change in feature |
| **MSE** | Average of squared errors |
| **RMSE** | Square root of MSE (interpretable units) |
| **MAE** | Average of absolute errors |
| **R²** | Percentage of variance explained by model |
| **Residual** | Difference between actual and predicted |
| **P-value** | Probability coefficient is actually zero |
| **OLS** | Method to find best-fitting line |

---

## 🎯 Final Tips

1. **Always visualize your data first** - plots reveal patterns statistics miss
2. **Check your residuals** - they tell you if your model assumptions hold
3. **Compare train and test scores** - to detect overfitting
4. **Focus on R² and RMSE** - most interpretable metrics
5. **P-value < 0.05** - your magic number for significance
6. **Higher correlation with target** - means better predictor

---

*Study Guide created for AIM Pillar 2 - Task 1: Housing Price Prediction*
