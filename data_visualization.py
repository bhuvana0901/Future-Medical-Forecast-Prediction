import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the cleaned data
df = pd.read_csv('cleaned_insurance.csv')

# Plot distributions of numerical features
sns.histplot(df['age'], kde=True)
plt.title('Age Distribution')
plt.show()

sns.histplot(df['bmi'], kde=True)
plt.title('BMI Distribution')
plt.show()

sns.histplot(df['charges'], kde=True)
plt.title('Charges Distribution')
plt.show()

# Plot the relationship between BMI and Charges
sns.scatterplot(x='bmi', y='charges', data=df)
plt.title('BMI vs. Charges')
plt.show()

# Plot the relationship between Age and Charges
sns.scatterplot(x='age', y='charges', data=df)
plt.title('Age vs. Charges')
plt.show()

# Plot the relationship between Children and Charges
sns.boxplot(x='children', y='charges', data=df)
plt.title('Children vs. Charges')
plt.show()

# Plot the distribution of Charges by Smoker status
sns.boxplot(x='smoker', y='charges', data=df)
plt.title('Smoker vs. Charges')
plt.show()

# Plot the distribution of Charges by Region
sns.boxplot(x='region_northwest', y='charges', data=df)
plt.title('Region (Northwest) vs. Charges')
plt.show()

sns.boxplot(x='region_southeast', y='charges', data=df)
plt.title('Region (Southeast) vs. Charges')
plt.show()

sns.boxplot(x='region_southwest', y='charges', data=df)
plt.title('Region (Southwest) vs. Charges')
plt.show()
