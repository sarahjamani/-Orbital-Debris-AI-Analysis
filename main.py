import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression

# ============================================
# LOAD NASA DATASET
# ============================================

file_path = "data/dataset.xlsx"

df = pd.read_excel(
    file_path,
    sheet_name="type_data",
    skiprows=2
)

# ============================================
# CLEAN DATASET
# ============================================

# Remove empty columns
df = df.dropna(axis=1, how='all')

# Keep first 17 useful columns
df = df.iloc[:, :17]

# Rename columns
df.columns = [
    'Year',
    'D1',
    'D2',
    'D3',
    'D4',
    'D5',
    'D6',
    'D7',
    'D8',
    'D9',
    'D10',
    'D11',
    'D12',
    'D13',
    'D14',
    'D15',
    'D16'
]

# ============================================
# CREATE TOTAL DEBRIS COLUMN
# ============================================

df['Total_Debris'] = df.iloc[:, 1:].sum(axis=1)

# ============================================
# CREATE AI LABELS
# ============================================

categories = []

for debris in df['Total_Debris']:

    if debris > 10000000:
        categories.append("Reusable")

    elif debris > 1000000:
        categories.append("Partially Recyclable")

    else:
        categories.append("Hazardous")

# Add labels to dataframe
df['Category'] = categories

# ============================================
# MACHINE LEARNING CLASSIFICATION
# ============================================

# Features
X = df[['D11', 'D7', 'D8']]

# Labels
y = df['Category']

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create AI model
classifier = DecisionTreeClassifier()

# Train model
classifier.fit(X_train, y_train)

# Make predictions
predictions = classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)

# ============================================
# PRINT AI RESULTS
# ============================================

print("\n===================================")
print("AI MODEL RESULTS")
print("===================================\n")

print("Training Data Size:", len(X_train))
print("Testing Data Size:", len(X_test))

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

print("\nSample Predictions:\n")

for i in range(5):

    print(
        f"Predicted: {predictions[i]} | Actual: {y_test.iloc[i]}"
    )

# ============================================
# REUSABILITY ANALYSIS
# ============================================

# Estimated reusable debris

d11_reusable = df['D11'].sum() * 0.40
d7_reusable = df['D7'].sum() * 0.30
d8_reusable = df['D8'].sum() * 0.20

# Total reusable debris
reusable = (
    d11_reusable +
    d7_reusable +
    d8_reusable
)

# Total debris
total_debris = df.iloc[:, 1:17].sum().sum()

# Non reusable debris
non_reusable = total_debris - reusable

# ============================================
# PRINT REUSABILITY RESULTS
# ============================================

print("\n===================================")
print("REUSABILITY ANALYSIS")
print("===================================\n")

print(f"D11 Reusable Contribution (40%): {d11_reusable:,.0f}")
print(f"D7 Reusable Contribution (30%): {d7_reusable:,.0f}")
print(f"D8 Reusable Contribution (20%): {d8_reusable:,.0f}")

print(f"\nTotal Reusable Debris: {reusable:,.0f}")
print(f"Total Non-Reusable Debris: {non_reusable:,.0f}")

# ============================================
# PIE CHART
# ============================================

labels = [
    'Reusable Debris',
    'Non-Reusable Debris'
]

sizes = [
    reusable,
    non_reusable
]

plt.figure(figsize=(8, 8))

plt.pie(
    sizes,
    labels=labels,
    autopct='%1.1f%%',
    startangle=90
)

plt.title(
    "Reusable vs Non-Reusable Orbital Debris",
    fontsize=16,
    fontweight='bold'
)

plt.show()

# ============================================
# REUSABILITY BAR CHART
# ============================================

debris_types = ['D11', 'D7', 'D8']
percentages = [40, 30, 20]

plt.figure(figsize=(8, 5))

bars = plt.bar(
    debris_types,
    percentages
)

plt.title(
    "Estimated Reusability of Major Debris Types",
    fontsize=16,
    fontweight='bold'
)

plt.xlabel("Debris Types")
plt.ylabel("Reusability Percentage (%)")

plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add values on bars
for bar in bars:

    yval = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        yval + 1,
        f"{yval}%",
        ha='center'
    )

plt.show()

# ============================================
# FUTURE DEBRIS PREDICTION
# ============================================

# Prepare prediction data
X_future = df[['Year']]
y_future = df['Total_Debris']

# Create prediction model
future_model = LinearRegression()

# Train model
future_model.fit(X_future, y_future)

# Future years
future_years = np.array([
    [2025],
    [2030],
    [2035],
    [2040],
    [2045],
    [2050]
])

# Predict future debris
future_predictions = future_model.predict(
    future_years
)

# ============================================
# FUTURE PREDICTION GRAPH
# ============================================

plt.figure(figsize=(10, 6))

# Current debris
plt.plot(
    df['Year'],
    df['Total_Debris'],
    label='Current Debris'
)

# Future prediction
plt.plot(
    future_years,
    future_predictions,
    linestyle='--',
    marker='o',
    label='Future Prediction'
)

plt.title(
    "Future Orbital Debris Prediction",
    fontsize=16,
    fontweight='bold'
)

plt.xlabel("Year")
plt.ylabel("Total Orbital Debris")

plt.legend()

plt.grid(True)

plt.show()

# ============================================
# AI CLASSIFICATION GRAPH
# ============================================

category_counts = df['Category'].value_counts()

plt.figure(figsize=(8, 5))

bars = plt.bar(
    category_counts.index,
    category_counts.values
)

plt.title(
    "AI Debris Classification Results",
    fontsize=16,
    fontweight='bold'
)

plt.xlabel("Debris Category")
plt.ylabel("Count")

plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add values on bars
for bar in bars:

    yval = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width()/2,
        yval + 1,
        int(yval),
        ha='center'
    )

plt.show()

# ============================================
# SUSTAINABILITY DASHBOARD
# ============================================

reusable_percent = (reusable / total_debris) * 100
non_reusable_percent = (non_reusable / total_debris) * 100

dashboard_labels = [
    'Reusable %',
    'Non-Reusable %'
]

dashboard_values = [
    reusable_percent,
    non_reusable_percent
]

plt.figure(figsize=(8, 5))

bars = plt.bar(
    dashboard_labels,
    dashboard_values
)

plt.title(
    "Orbital Debris Sustainability Dashboard",
    fontsize=16,
    fontweight='bold'
)

plt.ylabel("Percentage")

plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add values on bars
for bar in bars:

    yval = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width()/2,
        yval + 1,
        f"{yval:.1f}%",
        ha='center'
    )

plt.show()

# ============================================
# RECYCLING RECOMMENDATIONS
# ============================================

print("\n===================================")
print("RECYCLING RECOMMENDATIONS")
print("===================================\n")

recommendations = {
    "Reusable":
    "Metal recovery and spacecraft reuse recommended",

    "Partially Recyclable":
    "Partial recycling and material separation recommended",

    "Hazardous":
    "Safe orbital disposal required"
}

for category, recommendation in recommendations.items():

    print(f"{category} --> {recommendation}")

# ============================================
# EXPORT CLEANED DATASET
# ============================================

df.to_csv(
    "cleaned_space_debris_dataset.csv",
    index=False
)

print("\nCleaned dataset exported successfully!")