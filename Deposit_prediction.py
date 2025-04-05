#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# In[2]:


df = pd.read_csv('bank.csv')


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.shape


# In[6]:


df.describe()


# In[7]:


for col in df.select_dtypes(include='object').columns:
    print(col)
    print(df[col].unique())


# In[8]:


features_na = [features for features in df.columns if df[features].isnull().sum() > 0]
for features in features_na:
    print(features, np.round(df[features].isnull().mean(), 4), '% missing values')
else:
    print('No missing value is found')


# In[9]:


for column in df.columns:
    print(column, df[column].nunique())


# In[10]:


categorical_features = [feature for feature in df.columns if ((df[feature].dtypes == 'O') & (feature not in ['deposit']))]


# In[11]:


for feature in categorical_features:
    print(f'The feature is {feature} and the number of categories are {len(df[feature].unique())}')


# In[12]:


plt.figure(figsize=(15, 80), facecolor='white')
plotnumber = 1
for categorical_feature in categorical_features:
    ax = plt.subplot(12, 3, plotnumber)
    sns.countplot(y=categorical_feature, data=df)
    plt.xlabel(categorical_feature)
    plt.title(categorical_feature)
    plotnumber += 1
plt.show()


# In[13]:


for categorical_feature in categorical_features:
    sns.catplot(x='deposit', col=categorical_feature, kind='count', data=df)
plt.show()


# In[14]:


for categorical_feature in categorical_features:
    print(df.groupby(['deposit', categorical_feature]).size())


# In[15]:


numerical_features = [feature for feature in df.columns if ((df[feature].dtypes != 'O') & (feature not in ['deposit']))]
print('Number of numerical variables:', len(numerical_features))


# In[16]:


discrete_features = [feature for feature in numerical_features if len(df[feature].unique()) < 25]
print('Discrete variables count:', len(discrete_features))


# In[17]:


continuous_features = [feature for feature in numerical_features if feature not in discrete_features]
print('Continuous feature count:', len(continuous_features))


# In[18]:


plt.figure(figsize=(20, 60), facecolor='white')
plotnumber = 1
for continuous_feature in continuous_features:
    ax = plt.subplot(12, 3, plotnumber)
    sns.histplot(df[continuous_feature])
    plt.xlabel(continuous_feature)
    plotnumber += 1
plt.show()


# In[19]:


plt.figure(figsize=(20, 60), facecolor='white')
plotnumber = 1
for feature in continuous_features:
    ax = plt.subplot(12, 3, plotnumber)
    sns.boxplot(x='deposit', y=df[feature], data=df)
    plt.xlabel(feature)
    plotnumber += 1
plt.show()


# In[20]:


cor_mat = df.select_dtypes(include=['number']).corr()
plt.figure(figsize=(15, 7))
sns.heatmap(cor_mat, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.show()


# In[21]:


sns.countplot(x='deposit', data=df)
plt.show()


# In[22]:


df2 = df.copy()


# In[23]:


df2.drop(['default', 'pdays'], axis=1, inplace=True)


# In[24]:


df3 = df2[df2['campaign'] < 33]
df4 = df3[df3['previous'] < 31]


# In[25]:


cat_columns = ['job', 'marital', 'education', 'contact', 'month', 'poutcome']
for col in cat_columns:
    df4 = pd.concat([df4.drop(col, axis=1), pd.get_dummies(df4[col], prefix=col, drop_first=True)], axis=1)


# In[26]:


bool_columns = ['housing', 'loan', 'deposit']
for col in bool_columns:
    df4[col + '_new'] = df4[col].apply(lambda x: 1 if x == 'yes' else 0)
    df4.drop(col, axis=1, inplace=True)


# In[27]:


X = df4.drop(columns=['deposit_new'])
y = df4['deposit_new']


# In[28]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# In[29]:


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[30]:


models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42)
}


# In[31]:


results = {}
for model_name, model in models.items():
    if model_name == "Logistic Regression":
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    results[model_name] = {
        "Accuracy": accuracy,
        "Classification Report": classification_report(y_test, y_pred),
        "Confusion Matrix": confusion_matrix(y_test, y_pred)
    }


# In[32]:


for model_name, metrics in results.items():
    print(f"Model: {model_name}")
    print(f"Accuracy: {metrics['Accuracy']:.4f}")
    print("Classification Report:")
    print(metrics['Classification Report'])
    print("Confusion Matrix:")
    print(metrics['Confusion Matrix'])
    print("\n" + "="*50 + "\n")


# In[ ]:




