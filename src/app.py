from utils import db_connect
engine = db_connect()

# Import the libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV

# Load the dataset
bank_marketing_campaign_df = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/logistic-regression-project-tutorial/main/bank-marketing-campaign-data.csv', sep = ';')
bank_marketing_campaign_df

# information of the dataset
bank_marketing_campaign_df.info()
# Find the colums with unique values
bank_marketing_campaign_df.nunique()
# Find the colums with null values
bank_marketing_campaign_df.isnull().sum()
# We are not going to use the duration, campaign, pdays and previous columns because they are related to the last contact of the current campaign, so they are not useful for predicting the target variable
bank_marketing_campaign_df = bank_marketing_campaign_df.drop(['duration', 'campaign', 'pdays', 'previous'], axis = 1)
# Describe the dataframe
bank_marketing_campaign_df.describe(include='all')

# Visualize the categorical variables with a countplot
fig, ax = plt.subplots(4, 3, figsize=(30, 30))
# Countplot of the job column
sns.countplot(ax = ax[0,0], data = bank_marketing_campaign_df, x = 'job', palette="crest")
# Rotate x-axis labels
ax[0, 0].set_xticklabels(ax[0, 0].get_xticklabels(), rotation=15)
# Countplot of the marital column
sns.countplot(ax = ax[0,1], data = bank_marketing_campaign_df, x = 'marital', palette="crest")
# Countplot of the education column
sns.countplot(ax = ax[0,2], data = bank_marketing_campaign_df, x = 'education', palette="crest")
# Rotate x-axis labels
ax[0, 2].set_xticklabels(ax[0, 2].get_xticklabels(), rotation=15)
# Countplot of the default column
sns.countplot(ax = ax[1,0], data = bank_marketing_campaign_df, x = 'default', palette="crest")
# Countplot of the housing column
sns.countplot(ax = ax[1,1], data = bank_marketing_campaign_df, x = 'housing', palette="crest")
# Countplot of the loan column
sns.countplot(ax = ax[1,2], data = bank_marketing_campaign_df, x = 'loan', palette="crest")
# Countplot of the contact column
sns.countplot(ax = ax[2,0], data = bank_marketing_campaign_df, x = 'contact', palette="crest")
# Countplot of the month column
sns.countplot(ax = ax[2,1], data = bank_marketing_campaign_df, x = 'month', palette="crest")
# Countplot of the day_of_week column
sns.countplot(ax = ax[2,2], data = bank_marketing_campaign_df, x = 'day_of_week', palette="crest")
# Countplot of the poutcome column
sns.countplot(ax = ax[3,0], data = bank_marketing_campaign_df, x = 'poutcome', palette="crest")
# Countplot of the y column
sns.countplot(ax = ax[3,1], data = bank_marketing_campaign_df, x = 'y', palette="crest")
# Remove the empty subplots
fig.delaxes(ax[3,2])

# Display the plot
plt.show()

# Visualize the numerical variables with a histogram and boxplot
fig, ax = plt.subplots(2, 5, figsize=(30, 15))

# set the color palette
color = sns.color_palette("crest", 5)

# Histogram of the age column
sns.histplot(ax = ax[0,0], data = bank_marketing_campaign_df, x = 'age', color=color[0])
# Boxplot of the age column
sns.boxplot(ax = ax[1,0], data = bank_marketing_campaign_df, x = 'age', color=color[0])
# Histogram of the poutcome column
sns.histplot(ax = ax[0,1], data = bank_marketing_campaign_df, x = 'emp.var.rate', color=color[1])
# Boxplot of the poutcome column
sns.boxplot(ax = ax[1,1], data = bank_marketing_campaign_df, x = 'emp.var.rate', color=color[1])
# Histogram of the cons.price.idx column
sns.histplot(ax = ax[0,2], data = bank_marketing_campaign_df, x = 'cons.price.idx', color=color[2])
# Boxplot of the cons.price.idx column
sns.boxplot(ax = ax[1,2], data = bank_marketing_campaign_df, x = 'cons.price.idx', color=color[2])
# Histogram of the cons.conf.idx column
sns.histplot(ax = ax[0,3], data = bank_marketing_campaign_df, x = 'cons.conf.idx', color=color[3])
# Boxplot of the cons.conf.idx column
sns.boxplot(ax = ax[1,3], data = bank_marketing_campaign_df, x = 'cons.conf.idx', color=color[3])
# Histogram of the euribor3m column
sns.histplot(ax = ax[0,4], data = bank_marketing_campaign_df, x = 'euribor3m', color=color[4])
# Boxplot of the euribor3m column
sns.boxplot(ax = ax[1,4], data = bank_marketing_campaign_df, x = 'euribor3m', color=color[4])


# Display the plot
plt.show()

# For multivariate analysis, we are going to correlate all the variables with the target variable
# Relationship between the categorical variables and the target variable
fig, ax = plt.subplots(4, 3, figsize=(30, 30))

# set the colors
color = sns.color_palette("crest", 5)

# Countplot of the job column
sns.countplot(ax = ax[0,0], data = bank_marketing_campaign_df, x = 'job', hue = 'y', palette="crest")
# Rotate x-axis labels
ax[0, 0].set_xticklabels(ax[0, 0].get_xticklabels(), rotation=15)
# Countplot of the marital column
sns.countplot(ax = ax[0,1], data = bank_marketing_campaign_df, x = 'marital', hue = 'y', palette="crest")
# Countplot of the education column
sns.countplot(ax = ax[0,2], data = bank_marketing_campaign_df, x = 'education', hue = 'y', palette="crest")
# Rotate x-axis labels
ax[0, 2].set_xticklabels(ax[0, 2].get_xticklabels(), rotation=15)
# Countplot of the default column
sns.countplot(ax = ax[1,0], data = bank_marketing_campaign_df, x = 'default', hue = 'y', palette="crest")
# Countplot of the housing column
sns.countplot(ax = ax[1,1], data = bank_marketing_campaign_df, x = 'housing', hue = 'y', palette="crest")
# Countplot of the loan column
sns.countplot(ax = ax[1,2], data = bank_marketing_campaign_df, x = 'loan', hue = 'y', palette="crest")
# Countplot of the contact column
sns.countplot(ax = ax[2,0], data = bank_marketing_campaign_df, x = 'contact', hue = 'y', palette="crest")
# Countplot of the month column
sns.countplot(ax = ax[2,1], data = bank_marketing_campaign_df, x = 'month', hue = 'y', palette="crest")
# Countplot of the day_of_week column
sns.countplot(ax = ax[2,2], data = bank_marketing_campaign_df, x = 'day_of_week', hue = 'y', palette="crest")
# Countplot of the poutcome column
sns.countplot(ax = ax[3,0], data = bank_marketing_campaign_df, x = 'poutcome', hue = 'y', palette="crest")
# Remove the empty subplots
fig.delaxes(ax[3,1])
fig.delaxes(ax[3,2])

# Display the plot
plt.show()
# Now we are going to analyze the numerical variables
# Relationship between the numerical variables and the target variable
fig, ax = plt.subplots(2, 3, figsize=(30, 15))

# set the color palette
color = sns.color_palette("crest", as_cmap=True)

# Histogram of the age column
sns.histplot(ax = ax[0,0], data = bank_marketing_campaign_df, x = 'age', hue = 'y', palette="crest")
# Histogram of the poutcome column
sns.histplot(ax = ax[0,1], data = bank_marketing_campaign_df, x = 'emp.var.rate', hue = 'y', palette="crest")
# Histogram of the cons.price.idx column
sns.histplot(ax = ax[0,2], data = bank_marketing_campaign_df, x = 'cons.price.idx', hue = 'y', palette="crest")
# Histogram of the cons.conf.idx column
sns.histplot(ax = ax[1,0], data = bank_marketing_campaign_df, x = 'cons.conf.idx', hue = 'y', palette="crest")
# Histogram of the euribor3m column
sns.histplot(ax = ax[1,1], data = bank_marketing_campaign_df, x = 'euribor3m', hue = 'y', palette="crest")
# Remove the empty subplots
fig.delaxes(ax[1,2])

# Display the plot
plt.show()
# Feature engineering

# We need to group the categories basic.4y, basic.6y and basic.9y into a single category called basic
bank_marketing_campaign_df['education'] = bank_marketing_campaign_df['education'].replace(['basic.4y', 'basic.6y', 'basic.9y'], 'basic')

# Now, we need to eliminate the outliers of the age column
# Calculate the IQR
Q1 = bank_marketing_campaign_df['age'].quantile(0.25)
Q3 = bank_marketing_campaign_df['age'].quantile(0.75)
IQR = Q3 - Q1

# Eliminate the outliers
bank_marketing_campaign_df = bank_marketing_campaign_df[~((bank_marketing_campaign_df['age'] < (Q1 - 1.5 * IQR)) |(bank_marketing_campaign_df['age'] > (Q3 + 1.5 * IQR)))]
# Now, we are going to scale the variables with the MinMaxScaler

# Convert all the categorical variables into numerical variables with factorize method
bank_marketing_campaign_df['job_n'] = pd.factorize(bank_marketing_campaign_df['job'])[0]
bank_marketing_campaign_df['marital_n'] = pd.factorize(bank_marketing_campaign_df['marital'])[0]
bank_marketing_campaign_df['education_n'] = pd.factorize(bank_marketing_campaign_df['education'])[0]
bank_marketing_campaign_df['default_n'] = pd.factorize(bank_marketing_campaign_df['default'])[0]
bank_marketing_campaign_df['housing_n'] = pd.factorize(bank_marketing_campaign_df['housing'])[0]
bank_marketing_campaign_df['loan_n'] = pd.factorize(bank_marketing_campaign_df['loan'])[0]
bank_marketing_campaign_df['contact_n'] = pd.factorize(bank_marketing_campaign_df['contact'])[0]
bank_marketing_campaign_df['month_n'] = pd.factorize(bank_marketing_campaign_df['month'])[0]
bank_marketing_campaign_df['day_of_week_n'] = pd.factorize(bank_marketing_campaign_df['day_of_week'])[0]
bank_marketing_campaign_df['poutcome_n'] = pd.factorize(bank_marketing_campaign_df['poutcome'])[0]
bank_marketing_campaign_df['y_n'] = pd.factorize(bank_marketing_campaign_df['y'])[0]
numeric_columns = ['job_n', 'marital_n', 'education_n', 'default_n', 'housing_n', 'loan_n', 'contact_n',
                   'month_n', 'day_of_week_n', 'poutcome_n', 'y_n', 'age', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m']

# Create the scaler
scaler = MinMaxScaler()

# Scale the variables
scale_feature = scaler.fit_transform(bank_marketing_campaign_df[numeric_columns])
bank_marketing_campaign_scaled_df = pd.DataFrame(scale_feature, index = bank_marketing_campaign_df.index, columns = numeric_columns)
bank_marketing_campaign_scaled_df.head()

# Feature selection

# Separate the features and the target variable
X = bank_marketing_campaign_scaled_df.drop(['y_n'], axis = 1)
y = bank_marketing_campaign_scaled_df['y_n']

# Split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state=42)

# Select the best features with SelectKBest
best_features = SelectKBest(score_func = chi2, k = 10)
fit = best_features.fit(X, y)
ix = fit.get_support()

# Create a dataframe with the best features
X_test = pd.DataFrame(best_features.transform(X_test), index = X_test.index, columns = X_test.columns[ix])
X_train = pd.DataFrame(best_features.transform(X_train), index = X_train.index, columns = X_train.columns[ix])

# Display the dataframe
X_train.head()
X_test.head()
# We are going to add the target variable to the train and test dataframes
X_train['y_n'] = list(y_train)
X_test['y_n'] = list(y_test)

# Save the train and test dataframes
X_train.to_csv('../data/processed/X_train.csv', index = False)
X_test.to_csv('../data/processed/X_test.csv', index = False)
### **Step 3:** Build a Logistic Regression Model
# Read the processed data
train_data = pd.read_csv('../data/processed/X_train.csv')
test_data = pd.read_csv('../data/processed/X_test.csv')

# Display the train data
train_data.head()
# Display the test data
test_data.head()
# Separate the features and the target variable
X_train = train_data.drop(['y_n'], axis = 1)
y_train = train_data['y_n']
X_test = test_data.drop(['y_n'], axis = 1)
y_test = test_data['y_n']
# Initialization and training of the model
model = LogisticRegression()
model.fit(X_train, y_train)
# Model prediction
y_pred = model.predict(X_test)
y_pred
# acurracy of the model
accuracy_score(y_test, y_pred)
# Confusion matrix
campaign_cm = confusion_matrix(y_test, y_pred)

print(campaign_cm)

cm_df = pd.DataFrame(campaign_cm)

# Visualize the confusion matrix
plt.figure(figsize=(3,3))
sns.heatmap(cm_df, annot=True, fmt='d', cbar=False, cmap="crest")

# Display the plot
plt.tight_layout()
plt.show()

# Model optimization with hyperparameter tuning
# Create the hyperparameter grid
param_grid = {
    "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    "penalty": ["l1", "l2", "elasticnet", None],
    "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]
}

# Create the GridSearchCV object
grid = GridSearchCV(model, param_grid, cv = 5, verbose = 0, n_jobs = -1)
grid
# warning message off
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

# fit the model
grid.fit(X_train, y_train)

# best hyperparameters
print(f'Best hyperparameters: {grid.best_params_}')
# best model
model = LogisticRegression(C = 0.1, penalty = 'l1', solver = 'liblinear')
model.fit(X_train, y_train)
# Model prediction
y_pred = model.predict(X_test)
y_pred
# acurracy of the model
accuracy_score(y_test, y_pred)
# save the model
from pickle import dump
dump(model, open('../models/logistic_regression_C-0.1_penalty-l1_solver-liblinear_42.sav', 'wb'))
