## **Day 1: "Information Asymmetry and Market Failure"**
*Economic Concept: Lemon Market → Simple Rules → Basic Metrics*

```python
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

# Load and explore data
df = pd.read_csv('fraud_data.csv')
print(f"Dataset shape: {df.shape}")
print(f"Fraud rate: {df['is_fraud'].mean():.3f}")

# Economic Concept: Simple rule creates "lemon market"
def simple_rule_classifier(amount, threshold=500):
    """Economic intuition: Flag high amounts as suspicious"""
    return 1 if amount > threshold else 0

# Apply simple rule
df['prediction_day1'] = df['amt'].apply(lambda x: simple_rule_classifier(x, 500))

# Evaluate - introduce core metrics
def evaluate_model(y_true, y_pred, model_name):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print(f"\n{model_name} Results:")
    print(f"Precision: {precision:.3f} (Of flagged transactions, how many were actually fraud?)")
    print(f"Recall: {recall:.3f} (Of actual frauds, how many did we catch?)")
    print(f"F1-Score: {f1:.3f} (Harmonic mean of precision and recall)")
    
    return precision, recall, f1

# Day 1 results
day1_results = evaluate_model(df['is_fraud'], df['prediction_day1'], "Simple Rule ($500 threshold)")
```

## **Day 2: "Adverse Selection and Signaling Theory"**
*Economic Concept: Information Revelation → Decision Trees → Comparison*

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split

# Economic Concept: Trees reveal hidden information through "signals"
# Prepare features (market signals)
features = ['amt', 'city_pop', 'unix_time']  # Keep simple for visualization
X = df[features]
y = df['is_fraud']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Decision Tree - acts like market signaling mechanism
dt_classifier = DecisionTreeClassifier(max_depth=3, random_state=42)  # Limit depth for interpretability
dt_classifier.fit(X_train, y_train)

# Predictions
dt_predictions = dt_classifier.predict(X_test)

# Evaluate and compare with Day 1
day2_results = evaluate_model(y_test, dt_predictions, "Decision Tree (Market Signals)")

# Compare progress
print(f"\n📈 PROGRESS COMPARISON:")
print(f"Day 1 (Simple Rule): Precision={day1_results[0]:.3f}, Recall={day1_results[1]:.3f}")
print(f"Day 2 (Decision Tree): Precision={day2_results[0]:.3f}, Recall={day2_results[1]:.3f}")
print(f"Improvement in F1: {day2_results[2] - day1_results[2]:+.3f}")

# Visualize the "signaling mechanism"
plt.figure(figsize=(12, 8))
plot_tree(dt_classifier, feature_names=features, class_names=['Legitimate', 'Fraud'], filled=True)
plt.title("Economic Signaling Tree: How the Market Reveals Information")
plt.show()
```

## **Day 3: "Expected Value Under Uncertainty"**
*Economic Concept: Bayesian Decision Theory → Logistic Regression → ROC Analysis*

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Economic Concept: Probability-based decisions like rational economic actors
# Expand features for more sophisticated signaling
extended_features = ['amt', 'city_pop', 'unix_time', 'lat', 'long', 'merch_lat', 'merch_long']
X_extended = df[extended_features].fillna(0)  # Handle missing values
X_train_ext, X_test_ext, y_train, y_test = train_test_split(X_extended, y, test_size=0.3, random_state=42)

# Logistic Regression - mirrors rational economic decision making
lr_classifier = LogisticRegression(C=1.0, random_state=42)
lr_classifier.fit(X_train_ext, y_train)

# Get probabilities (expected values)
lr_probabilities = lr_classifier.predict_proba(X_test_ext)[:, 1]
lr_predictions = lr_classifier.predict(X_test_ext)

# Evaluate
day3_results = evaluate_model(y_test, lr_predictions, "Logistic Regression (Expected Value)")

# ROC Curve - evaluating across all possible thresholds
fpr, tpr, thresholds = roc_curve(y_test, lr_probabilities)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve: Expected Value Optimization')
plt.legend()

# Compare all three days
plt.subplot(1, 2, 2)
models = ['Day 1\n(Simple Rule)', 'Day 2\n(Decision Tree)', 'Day 3\n(Logistic Reg)']
f1_scores = [day1_results[2], day2_results[2], day3_results[2]]
plt.bar(models, f1_scores, color=['red', 'orange', 'green'])
plt.title('Economic Evolution: F1-Score Progress')
plt.ylabel('F1-Score')
plt.show()

print(f"\n📊 THREE-DAY PROGRESS:")
for i, (model, f1) in enumerate(zip(models, f1_scores), 1):
    print(f"Day {i}: {f1:.3f}")
```

## **Day 4: "Preventing Gresham's Law"**
*Economic Concept: Anti-Gaming → Random Forest → Grid Search*

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Economic Concept: Multiple diverse models prevent systematic gaming
# Expand feature set further (diverse "auditors")
all_features = ['amt', 'city_pop', 'unix_time', 'lat', 'long', 'merch_lat', 'merch_long']
# Add engineered features (like multiple independent auditors)
df['hour'] = pd.to_datetime(df['trans_time']).dt.hour
df['amount_log'] = np.log1p(df['amt'])  # Log transform for better distribution

final_features = all_features + ['hour', 'amount_log']
X_final = df[final_features].fillna(0)
X_train_final, X_test_final, y_train, y_test = train_test_split(X_final, y, test_size=0.3, random_state=42)

# Grid Search - finding optimal "auditor committee"
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [3, 5, 7]
}

rf_classifier = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf_classifier, param_grid, cv=3, scoring='f1', n_jobs=-1)
grid_search.fit(X_train_final, y_train)

# Best model
best_rf = grid_search.best_estimator_
rf_predictions = best_rf.predict(X_test_final)

day4_results = evaluate_model(y_test, rf_predictions, "Random Forest (Anti-Gaming)")

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best CV Score: {grid_search.best_score_:.3f}")

# Feature importance - which "signals" matter most
feature_importance = pd.DataFrame({
    'feature': final_features,
    'importance': best_rf.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n🔍 MARKET SIGNALS IMPORTANCE:")
print(feature_importance.head())

# Progressive comparison
all_results = [day1_results[2], day2_results[2], day3_results[2], day4_results[2]]
days = ['Day 1', 'Day 2', 'Day 3', 'Day 4']

plt.figure(figsize=(10, 6))
plt.plot(days, all_results, marker='o', linewidth=2, markersize=8)
plt.title('Economic Learning Curve: F1-Score Evolution')
plt.ylabel('F1-Score')
plt.grid(True, alpha=0.3)
plt.show()
```

## **Day 5: "Market Efficiency and Optimization"**
*Economic Concept: Pareto Efficiency → Model Selection → Multi-Objective Optimization*

```python
from sklearn.metrics import classification_report, roc_auc_score

# Economic Concept: Find Pareto-optimal solutions (can't improve one without worsening another)
def comprehensive_evaluation(models_dict, X_test, y_test):
    """Evaluate all models across multiple metrics - finding Pareto frontier"""
    results = {}
    
    for name, (model, predictions) in models_dict.items():
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X_test)[:, 1]
            auc_score = roc_auc_score(y_test, proba)
        else:
            auc_score = None
            
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        
        results[name] = {
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'AUC': auc_score
        }
    
    return pd.DataFrame(results).T

# Recreate all models for final comparison (using previous day's data splits)
models_comparison = {
    'Simple Rule': (None, df.loc[X_test.index, 'prediction_day1']),
    'Decision Tree': (dt_classifier, dt_predictions),
    'Logistic Regression': (lr_classifier, lr_predictions),
    'Random Forest': (best_rf, rf_predictions)
}

# Comprehensive evaluation
final_results = comprehensive_evaluation(models_comparison, X_test_final, y_test)
print("🏆 FINAL ECONOMIC EFFICIENCY COMPARISON:")
print(final_results.round(3))

# Pareto Frontier Visualization
plt.figure(figsize=(12, 8))

# Precision vs Recall (classic trade-off)
plt.subplot(2, 2, 1)
for model in final_results.index:
    plt.scatter(final_results.loc[model, 'Recall'], 
               final_results.loc[model, 'Precision'], 
               label=model, s=100)
plt.xlabel('Recall (Fraud Detection Rate)')
plt.ylabel('Precision (Alert Accuracy)')
plt.title('Pareto Frontier: Precision vs Recall')
plt.legend()

# F1 Score evolution
plt.subplot(2, 2, 2)
plt.bar(final_results.index, final_results['F1-Score'], color=['red', 'orange', 'green', 'blue'])
plt.title('Economic Efficiency: F1-Score Comparison')
plt.xticks(rotation=45)

# AUC comparison (where available)
plt.subplot(2, 2, 3)
auc_data = final_results.dropna(subset=['AUC'])
plt.bar(auc_data.index, auc_data['AUC'], color=['orange', 'green', 'blue'])
plt.title('Area Under Curve: Overall Performance')
plt.xticks(rotation=45)

# Economic lesson summary
plt.subplot(2, 2, 4)
lessons = ['Information\nAsymmetry', 'Signaling\nTheory', 'Expected\nValue', 'Anti-Gaming', 'Pareto\nEfficiency']
concepts = ['Rule-based', 'Decision Tree', 'Logistic Reg', 'Random Forest', 'Model Selection']
plt.barh(range(len(lessons)), [0.2, 0.4, 0.6, 0.8, 1.0], color='lightblue', alpha=0.7)
for i, (lesson, concept) in enumerate(zip(lessons, concepts)):
    plt.text(0.1, i, f"{lesson}\n→ {concept}", ha='left', va='center', fontsize=9)
plt.title('Economic Concepts → ML Evolution')
plt.xlim(0, 1.2)
plt.yticks([])

plt.tight_layout()
plt.show()

# Final economic insights
print(f"\n💡 ECONOMIC INSIGHTS:")
best_model = final_results['F1-Score'].idxmax()
print(f"• Most economically efficient model: {best_model}")
print(f"• Key insight: {final_results.loc[best_model, 'F1-Score']:.3f} F1-score represents optimal trade-off")
print(f"• Business impact: Moved from {final_results.iloc[0]['F1-Score']:.3f} to {final_results.loc[best_model, 'F1-Score']:.3f}")
```

## **Teaching Notes for Each Day:**

**Pedagogical Connections:**
- **Day 1→2**: "Simple rules create information asymmetry. How can we reveal hidden patterns?"
- **Day 2→3**: "Fixed decisions are rigid. How do rational actors use probabilities?"
- **Day 3→4**: "Single models can be gamed. How do we prevent systematic exploitation?"
- **Day 4→5**: "Multiple good models exist. How do we choose the economically optimal one?"

Each day's code builds on the previous day's `X_test` and `y_test` sets, allowing direct comparison and demonstrating economic progression through ML sophistication.