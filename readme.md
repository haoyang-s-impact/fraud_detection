## **5-Day Data Science Course: "Economics of Fraud Detection"**
**Target**: High school students | **Format**: 5 × 1.5 hours | **Structure**: 15 min economics + 30 min visualization + 45 min ML coding

---

### **Day 1: "Information Asymmetry and Market Failure"**
**Economic Phenomenon**: **Lemon Market Theory** - When buyers can't distinguish quality, markets fail

**Connection to Model**: Fraudsters have perfect information (they know they're frauding), but our system has imperfect information → Simple rule-based systems create lemon markets where we can't distinguish good from bad transactions

**Visualization Development (30 mins)**: Daily Revenue Risk Dashboard showing losses from missed fraud vs. revenue loss from blocking legitimate customers

**ML Code Development (45 mins)**:
- Simple rule-based classifier: "Flag if amount > $X"
- **Evaluation Focus**: Introduction to precision, recall, F1-score
- **Why These Metrics**: In lemon markets, we need to measure both "catching bad actors" (recall) and "not alienating good customers" (precision)

**Advancement**: Rules work but create too many false positives → Need better information revelation mechanisms

---

### **Day 2: "Adverse Selection and Signaling Theory"**
**Economic Phenomenon**: Markets develop mechanisms to reveal hidden information through observable signals

**Connection to Model**: Decision trees act like market signaling mechanisms - they find distinguishing characteristics (transaction patterns) that reveal hidden information about fraud likelihood

**Visualization Development (30 mins)**: Customer Risk Profile Heat Map showing which behavioral signals correlate with fraud

**ML Code Development (45 mins)**:
- Decision tree implementation with single hyperparameter tuning (max_depth)
- **Evaluation Focus**: Direct precision/recall comparison with Day 1 model
- **Why This Advance**: Shows how information revelation (tree splits) improves our ability to distinguish good from bad

**Advancement**: Fixed thresholds still limit us → Need probability-based decisions

---

### **Day 3: "Expected Value Under Uncertainty"**  
**Economic Phenomenon**: **Bayesian Decision Theory** - Rational actors make optimal decisions considering both probability and consequences

**Connection to Model**: Logistic regression mirrors how rational economic actors should decide - by considering probability of fraud AND the costs of different mistakes

**Visualization Development (30 mins)**: Investigation ROI Calculator showing expected value calculations for different decision thresholds

**ML Code Development (45 mins)**:
- Logistic regression with single hyperparameter tuning (regularization C)
- **Evaluation Focus**: ROC curves and AUC - measuring performance across ALL possible thresholds
- **Why This Advance**: Moves from fixed rules to probability-based expected value optimization

**Advancement**: Single models can be systematically fooled → Need diverse perspectives

---

### **Day 4: "Preventing Gresham's Law"**
**Economic Phenomenon**: **"Bad money drives out good money"** when you can't distinguish between them

**Connection to Model**: Sophisticated fraudsters can learn to game single models (bad transactions driving out good detection). Ensemble methods prevent this by making it much harder to fool multiple diverse models simultaneously

**Visualization Development (30 mins)**: Alert Prioritization Dashboard comparing single model vs. ensemble effectiveness

**ML Code Development (45 mins)**:
- Random Forest implementation
- **Evaluation Focus**: Grid search introduction (multiple hyperparameters: n_estimators + max_depth)
- **Why This Advance**: Multiple models prevent systematic gaming, like having multiple independent auditors

**Advancement**: Now we have multiple models → Need framework to choose optimal solution

---

### **Day 5: "Market Efficiency and Optimization"**
**Economic Phenomenon**: **Pareto Efficiency** - Finding solutions where you can't improve one outcome without worsening another

**Connection to Model**: Model selection finds the economically optimal point on the precision-recall curve - the Pareto frontier where we can't reduce false positives without missing more fraud

**Visualization Development (30 mins)**: Monthly Impact Projection showing business trade-offs and optimal operating points

**ML Code Development (45 mins)**:
- Comprehensive model comparison framework across all 4 models
- **Evaluation Focus**: Multi-metric optimization finding Pareto-optimal solutions
- **Why This Culmination**: Business decisions require finding economically efficient trade-offs, not just maximizing single metrics

---

## **Progressive Learning Architecture**

**Economic Concept Progression**: 
Information Asymmetry → Signaling → Expected Value → Anti-Gaming → Optimization

**Model Complexity Progression**: 
Simple Rules → Decision Trees → Logistic Regression → Random Forest → Model Selection

**Evaluation Sophistication Progression**: 
Basic Metrics → Direct Comparison → ROC Analysis → Grid Search → Multi-Objective Optimization

**Key Pedagogical Connections**:
1. **Economics → Model**: Each economic problem directly motivates why the previous model fails and what the new model solves
2. **Model → Evaluation**: Each model advancement requires more sophisticated evaluation methods
3. **Evaluation → Next Economics**: Each evaluation reveals new limitations, leading to the next economic problem
