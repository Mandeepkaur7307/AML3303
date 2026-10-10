# AML 3303 Assessment 1: Delivery Delay Prediction & Slack Alerts

**GitHub Repository:** https://github.com/Mandeepkaur7307/AML3303

## 1. Project Overview

### Business Problem

MapleFreight Logistics transports approximately 8,000 shipments per week across Ontario, Quebec, and the Prairies. Its on-time delivery performance has declined from 94% to 88%. Late deliveries can lead to service credits, additional delivery expenses, and customer dissatisfaction.

This project develops a machine learning system to predict whether a shipment is likely to be delivered late. It also integrates Slack notifications to alert the dispatch team about high-risk shipments so that they can investigate potential problems and take preventive action.

### Project Objectives

- Analyze shipment data to identify patterns associated with late deliveries.
- Clean the dataset and handle missing values and duplicate records.
- Perform exploratory data analysis (EDA).
- Prepare numerical and categorical features for machine learning.
- Train and compare classification models.
- Evaluate models using accuracy, precision, recall, F1-score, and confusion matrices.
- Identify important predictive features and operational risk patterns.
- Integrate Slack notifications for high-risk shipments.
- Provide recommendations to help improve delivery operations.

## 2. Technologies Used

- **Programming language:** Python
- **Data analysis:** Pandas, NumPy
- **Data visualization:** Matplotlib, Seaborn
- **Machine learning:** Scikit-learn
- **Slack integration:** Slack Incoming Webhook, Requests
- **Environment configuration:** Python-dotenv
- **Development environment:** Jupyter Notebook / VS Code
- **Version control:** Git and GitHub

The prediction system uses traditional machine learning classification algorithms. It does not use a large language model (LLM) to make delivery-delay predictions.

## 3. Dataset Description

**Dataset file:** `maplefreight_delivery_delay_dataset (1).csv`

The original dataset contained 6,035 rows. After removing 32 duplicate rows, the cleaned dataset contained 6,003 rows.

The dataset includes shipment-related information such as:

- Shipment month and distance
- Shipment weight and number of stops
- Cross-border shipment status
- Scheduled transit time
- Pickup delay
- Driver experience and vehicle age
- Traffic index and route congestion
- Previous late deliveries
- Fuel cost
- Weather conditions
- Carrier information
- Actual transit time
- Delivery outcome

### Target Variable

The target variable is `delivered_late`.

- `0` = Shipment delivered on time.
- `1` = Shipment delivered late.

This is a binary classification problem because the model predicts one of two possible outcomes.

### Target Distribution After Deduplication

| Delivery status | Count | Percentage |
|---|---:|---:|
| On-time shipments (0) | 4,768 | 79.43% |
| Late shipments (1) | 1,235 | 20.57% |
| **Total** | **6,003** | **100%** |

Approximately one in five shipments in the cleaned dataset was late. Since on-time shipments are the majority class, accuracy alone is not sufficient to evaluate the model.

## 4. Data Cleaning and Preprocessing

### Data Cleaning

The initial dataset was inspected for duplicate records, missing values, data types, target distribution, and unusual numerical values.

The following cleaning steps were performed:

1. Removed 32 duplicate rows, reducing the dataset from 6,035 to 6,003 rows.
2. Checked repeated shipment identifiers; three repeated shipment IDs were reported and require review to determine whether they represent valid repeated records or conflicting shipments.
3. Investigated missing values in `driver_experience_years`, `traffic_index`, and `fuel_cost_cad`.
4. Checked numerical summaries to identify unusual values and potential outliers.
5. Separated the target variable from the model input features.
6. Prepared numerical and categorical features for model training.

### Missing-Value Handling

The numerical columns were prepared using median imputation, while categorical columns were prepared using most-frequent-value imputation.

The reported descriptive statistics show the following counts:

| Feature | Available values | Missing values after deduplication |
|---|---:|---:|
| `driver_experience_years` | 5,793 | 210 |
| `traffic_index` | 5,883 | 120 |
| `fuel_cost_cad` | 5,763 | 240 |

These counts are based on the reported descriptive statistics before imputation.

### Feature Preprocessing

The machine learning preprocessing pipeline uses:

- **Numerical features:** Median imputation and standardization where required.
- **Categorical features:** Most-frequent imputation and one-hot encoding.
- **Train/test split:** An 80/20 split with stratification.
- **Identifier handling:** `shipment_id` is excluded because it is an identifier rather than a useful operational measurement.

The feature `actual_transit_hours` is excluded from prediction because it describes the completed journey and may introduce data leakage. At prediction time, the model should only use information available while the shipment is still in transit.

Unusual values, such as the reported maximum weight of 2,999,300 kg and maximum driver experience of 35 years, should be checked against the dataset's definitions and units before deployment. Extreme values should not be removed automatically without investigation.

## 5. Exploratory Data Analysis Findings

### Late-Delivery Rates by Carrier

The following late-delivery rates were calculated from the dataset:

| Carrier | Observed late-delivery rate |
|---|---:|
| Contract Owner-Op | 35.90% |
| PrairieLine | 26.50% |
| QuebecExpress | 22.40% |
| NorthStar Carriers | 21.99% |
| MapleFreight Fleet | 12.88% |

Contract Owner-Op had the highest observed late-delivery rate among the listed carriers, at approximately 35.90%. MapleFreight Fleet had the lowest listed rate, at approximately 12.88%.

These differences may help identify areas for further investigation. However, carrier rates should also be considered alongside shipment volume, route difficulty, distance, weather, and other operating conditions. The rates alone do not establish that a carrier causes delays.

### Late-Delivery Rates by Weather Condition

| Weather condition | Observed late-delivery rate |
|---|---:|
| Storm | 46.32% |
| Snow | 33.49% |
| Fog | 21.28% |
| Rain | 20.87% |
| Clear | 14.05% |

Storm conditions had the highest observed late-delivery rate, followed by snow. Clear weather had the lowest rate among the listed conditions.

This pattern suggests that weather conditions may be useful for identifying shipments requiring additional attention. The analysis describes an association, not proof of causation.

The number of shipments in each category should be checked before making operational decisions based on these percentages.

### Numerical Data Observations

The descriptive statistics show an average shipment distance of approximately 632.13 km and an average scheduled transit time of approximately 10.50 hours.

The average pickup delay was approximately 20.35 minutes. The median pickup delay was -6 minutes, indicating that at least some pickups occurred earlier than the recorded reference time.

The dataset also contains missing operational values and unusually large numerical observations. These should be validated to avoid data-quality problems affecting model performance.

## 6. Machine Learning Models

Three approaches were compared to evaluate the prediction system.

### Majority Baseline

The majority baseline predicts the most common class. It provides a reference point to determine whether the classification models identify late shipments more effectively than a simple majority-class prediction.

### Logistic Regression

Logistic Regression estimates the probability that a shipment will be delivered late. It provides a useful classification baseline and produces probability estimates that can be used to flag potentially delayed shipments.

### Random Forest

Random Forest combines multiple decision trees to learn relationships between shipment characteristics and delivery outcomes. It can capture nonlinear patterns and provide feature-importance estimates.

### Evaluation Metrics

- **Accuracy:** The proportion of all predictions that are correct.
- **Precision:** The proportion of predicted late shipments that were actually late.
- **Recall:** The proportion of actual late shipments correctly identified.
- **F1-score:** The harmonic mean of precision and recall.
- **Confusion matrix:** Summarizes correct and incorrect predictions for each class.

Recall is particularly important for this project because a missed late shipment may prevent the dispatch team from taking timely action. Precision also matters because too many false alerts can create unnecessary work.

## 7. Model Comparison Results

The following metrics were obtained from the reported model evaluation:

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Majority Baseline | 0.794 | 0.000 | 0.000 | 0.000 |
| Logistic Regression | 0.749 | 0.436 | 0.745 | 0.550 |
| Random Forest | 0.808 | 0.870 | 0.081 | 0.148 |

The test set contained 1,201 observations: 954 on-time shipments and 247 late shipments.

### Logistic Regression Results

The Logistic Regression model achieved:

- Accuracy: 0.749
- Precision for late shipments: 0.436
- Recall for late shipments: 0.745
- F1-score for late shipments: 0.550

Its confusion matrix was:

```text
[[716 238]
 [ 63 184]]
```

This means:

- 716 on-time shipments were correctly classified as on time.
- 238 on-time shipments were incorrectly flagged as late.
- 63 late shipments were incorrectly predicted to be on time.
- 184 late shipments were correctly identified.

The model detected 184 of the 247 actual late shipments in the test set.

### Random Forest Results

The Random Forest model achieved:

- Accuracy: 0.808
- Precision for late shipments: 0.870
- Recall for late shipments: 0.081
- F1-score for late shipments: 0.148

Its reported confusion matrix was:

```text
[[951   3]
 [227  20]]
```

This means:

- 951 on-time shipments were correctly classified.
- 3 on-time shipments were incorrectly flagged as late.
- 227 late shipments were incorrectly predicted to be on time.
- 20 late shipments were correctly identified.

Although Random Forest achieved the highest accuracy and precision, it identified only 20 of the 247 late shipments at the evaluated settings.

### Model Selection

Logistic Regression is the initial candidate for the delay-alert system because it achieved substantially higher recall and a higher F1-score than Random Forest.

The business objective is to identify late shipments early, so missing fewer late shipments is an important consideration. The final model and alert threshold should be selected using validation results and the operational costs of false alerts and missed delays.

The model comparison demonstrates why the highest accuracy does not necessarily indicate the best model for an imbalanced classification problem.

## 8. Feature Importance

The Random Forest feature-importance analysis identified the following leading features:

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | Scheduled transit hours | 0.083278 |
| 2 | Traffic index | 0.079868 |
| 3 | Distance in kilometres | 0.071860 |
| 4 | Pickup delay in minutes | 0.067071 |
| 5 | Route congestion score | 0.062011 |
| 6 | Driver experience in years | 0.059088 |
| 7 | Vehicle age in years | 0.053599 |
| 8 | Shipment weight | 0.053310 |
| 9 | Fuel cost (CAD) | 0.051405 |
| 10 | Number of stops | 0.041085 |

Scheduled transit time had the highest reported feature importance, followed by traffic index and distance.

These features may help the model distinguish between shipments with different delivery risks. However, feature importance does not indicate whether a feature increases or decreases delay risk, and it does not establish a causal relationship.

The importance rankings should also be interpreted in light of data quality, feature availability at prediction time, and possible correlations between variables.

## 9. Slack Alert Integration

### Purpose

Slack integration provides a communication mechanism for notifying the dispatch team about shipments identified as high risk by the model.

### Configuration

The project uses a Slack Incoming Webhook to send messages to a configured Slack channel.

The webhook URL is stored in a local `.env` file:

```text
SLACK_WEBHOOK_URL=your_real_slack_webhook_url_here
```

The actual webhook URL must remain private and must never be committed to GitHub.

The `.gitignore` file should exclude `.env`, virtual environments, Python cache files, and notebook checkpoint files. The `.env.example` file should contain only a placeholder value.

### Alert Workflow

The intended alert workflow is:

1. Load the Slack webhook URL from the environment.
2. Load the trained classification model.
3. Obtain the current shipment records eligible for prediction.
4. Calculate the probability of late delivery for each shipment.
5. Compare each probability with the configured risk threshold.
6. Sort qualifying shipments by risk score.
7. Select up to five high-risk shipments for notification.
8. Send the alert to Slack.
9. Handle errors and the situation in which no shipments meet the threshold.

The alert should include the shipment ID, destination, carrier, and predicted risk score when these fields are available.

### Alert Threshold

The example implementation uses a threshold of 0.40, meaning that shipments with a predicted late-delivery probability of at least 40% qualify for an alert.

This threshold is a configurable starting point, not a confirmed optimal threshold. It should be selected using validation data by considering recall, precision, the number of alerts generated, and the cost of missed delays.

The test set should be reserved for final model evaluation rather than repeatedly used to tune the threshold.

### Slack Evidence

A screenshot of the actual Slack alert should be saved at:

```text
screenshots/slack.png
```

The screenshot should show a successfully received message in the intended Slack channel. Only include this evidence after the webhook has been tested successfully.

## 10. Installation and Execution

### Prerequisites

Install Python and Git. Use Jupyter Notebook or VS Code to run the project.

### Step 1: Clone the Repository

```bash
git clone https://github.com/Mandeepkaur7307/AML3303.git
cd AML3303
```

### Step 2: Create a Virtual Environment (Recommended)

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Step 3: Install Dependencies

If the repository contains `requirements.txt`, run:

```bash
python -m pip install -r requirements.txt
```

The project uses the libraries required by its implemented code, including Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Requests, and Python-dotenv where applicable.

### Step 4: Configure the Dataset

Place the CSV in the location expected by the notebook, or update the dataset path in the code to match the local file.

### Step 5: Configure Slack

Create a local `.env` file with your own Incoming Webhook URL. Keep the file private and verify that Git ignores it.

### Step 6: Run the Notebook

Open the notebook in Jupyter Notebook or VS Code. Run the cells in order to reproduce the data cleaning, EDA, preprocessing, model training, evaluation, and Slack integration tests.

The notebook should be tested from a fresh kernel to confirm that the workflow does not depend on variables created manually during an earlier session.

## 11. Recommendations

Based on the current analysis, the following actions are recommended:

1. **Use Logistic Regression as the initial candidate.** Its recall of 0.745 is more suitable for a system focused on detecting late shipments than the evaluated Random Forest model.

2. **Investigate high-risk weather conditions.** Storm and snow shipments had the highest observed late-delivery rates in the reported analysis. Dispatch planning could account for these conditions.

3. **Review carrier performance.** Contract Owner-Op had the highest observed carrier late-delivery rate in the supplied results. Further investigation should account for shipment volumes and route differences.

4. **Monitor traffic and route congestion.** These features ranked highly in the Random Forest feature-importance results and may help identify shipments needing attention.

5. **Validate the alert threshold.** Select a threshold using validation data that balances missed delays against unnecessary alerts.

6. **Monitor operational performance.** Track model precision, recall, alert volume, false positives, false negatives, and changes in shipment patterns over time.

7. **Protect operational data.** Keep webhook credentials private and limit access to shipment information and Slack alerts.

## 12. Limitations

The current results represent an evaluation on the available dataset and do not guarantee the same performance in live operations.

The observed carrier and weather rates may be affected by differences in shipment volume, route characteristics, and other factors. Feature importance also describes predictive usefulness rather than causation.

Before deployment, the system should be tested using realistic in-transit shipment data, validated for data leakage, and assessed for performance on future shipments. The alert threshold and number of alerts should be evaluated against the dispatch team's operational needs.

## 13. AI Usage Disclosure

Document any AI assistance according to the course's academic integrity requirements. If AI tools were used to explain code, troubleshoot errors, or organize documentation, describe that assistance accurately.

The prediction workflow described in this README uses traditional machine learning algorithms rather than an LLM to predict delivery delays. All reported results should be verified against the executed notebook.

## Conclusion

This project demonstrates a machine learning workflow for predicting delivery delays and communicating high-risk shipments through Slack.

The cleaned dataset contains 6,003 shipment records, of which 1,235 are labeled late. Logistic Regression achieved a recall of 0.745 and an F1-score of 0.550, while Random Forest achieved higher accuracy and precision but substantially lower recall at the evaluated settings.

The results support using Logistic Regression as the initial candidate for a delay-detection workflow. Before operational deployment, the system requires validation of the alert threshold, testing on realistic in-transit records, review of data quality, and confirmation that Slack alerts work reliably.

The notebook, README, dependency list, configuration example, and Slack screenshot provide the documentation and evidence needed to reproduce and assess the project.