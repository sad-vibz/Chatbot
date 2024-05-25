# Simple Chatbot Project

This project involves creating a simple chatbot using a dataset. The notebook provides a step-by-step guide to load, clean, and analyze the data, as well as handling missing values and checking for duplicate and inconsistent entries.

## Project Structure

The notebook is structured into the following sections:

1. **Loading the Dataset**
2. **Exploratory Data Analysis (EDA)**
3. **Data Cleaning**
4. **Handling Missing Values**
5. **Detecting Duplicate and Inconsistent Entries**

## Getting Started

### Prerequisites

- Python 3.x
- Jupyter Notebook
- pandas

### Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/simple-chatbot-project.git
2.	Navigate to the project directory:
3.	cd simple-chatbot-project
4.	Install the required packages:
5.	pip install pandas jupyter
Usage
1.	Launch Jupyter Notebook:
2.	jupyter notebook
3.	Open the Simple Chatbot Project.ipynb notebook.
4.	Run the cells sequentially to execute the code and follow along with the comments and explanations.
Code Explanation
Loading the Dataset:
import pandas as pd
df = pd.read_csv('Chatbot_Dataset.csv')
print("Head:\n", df.head())
print("\nTail:\n", df.tail())
print("\nInfo:\n", df.info())
print("\nDescribe:\n", df.describe())
Handling Missing Values:
null_values = df.isnull().sum()
print("Null Values Before Handling:")
print(null_values)
numerical_cols = df.select_dtypes(include=['number']).columns
non_numerical_cols = df.select_dtypes(exclude=['number']).columns
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())
for col in non_numerical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])
df.dropna(inplace=True)
print("\nNull Values After Handling:")
print(df.isnull().sum())
Detecting Duplicate and Inconsistent Entries:
duplicate_entries = df.duplicated().sum()
print("Number of duplicate entries:", duplicate_entries)
category_values = df['category']
intent_values = df['intent']
inconsistent_entries = (category_values > intent_values).sum()
print("Number of inconsistent entries:", inconsistent_entries)
Contributing
Contributions are welcome! Please fork the repository and create a pull request with your changes.
License
This project is licensed under the MIT License.
Acknowledgements
•	The dataset used for this project.
•	The libraries and tools that made this project possible.

