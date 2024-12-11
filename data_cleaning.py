import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load raw data
df = pd.read_csv('insurance.csv')

# Data cleaning
df.drop_duplicates(inplace=True)

# Encoding categorical features
label_encoder = LabelEncoder()
df['sex'] = label_encoder.fit_transform(df['sex'])
df['smoker'] = label_encoder.fit_transform(df['smoker'])
df = pd.get_dummies(df, columns=['region'], drop_first=True)

# Scaling numerical features
scaler = StandardScaler()
df[['age', 'bmi', 'children']] = scaler.fit_transform(df[['age', 'bmi', 'children']])

# Save the cleaned data to a CSV file
df.to_csv('cleaned_data.csv', index=False)
