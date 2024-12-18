# preprocessing.py
import numpy as np
import pandas as pd

# =====================================================================Functions for data pre-processing========================================================================

## Additional : Data information fsunction
# Checking basic data information
def check_data_information(data, cols):
    list_item = []
    for col in cols:
        # Convert unique values to string representation
        unique_sample = ', '.join(map(str, data[col].unique()[:5]))
        
        list_item.append([
            col,                                           # The column name
            str(data[col].dtype),                          # The data type as string
            data[col].isna().sum(),                        # The count of null values
            round(100 * data[col].isna().sum() / len(data[col]), 2),  # The percentage of null values
            data.duplicated().sum(),                       # The count of duplicated rows
            data[col].nunique(),                           # The count of unique values
            unique_sample                                  # Sample of unique values as string
        ])

    desc_df = pd.DataFrame(
        data=list_item,
        columns=[
            'Feature',
            'Data Type',
            'Null Values',
            'Null Percentage',
            'Duplicated Values',
            'Unique Values',
            'Unique Sample'
        ]
    )
    return desc_df

# Initial data transformation
def initial_data_transform(data):
    # Rename some columns
    data = data.rename(columns={'CURRENT_JOB_YRS' : 'Current_Job_Years',
                                'CURRENT_HOUSE_YRS' : 'Current_House_Years',
                                'CITY' : 'City',
                                'STATE' : 'State',
                                'Married/Single' : 'Marital_Status'})
    
    # Clean invalid characters and lowercase the values of categorical columns
    columns_to_clean = ['Profession', 'State', 'City']

    # Removing those characters from the 'Profession', 'City', and 'State' column
    for col in columns_to_clean:
        data[col] = data[col].str.replace(r'\[\d+\]', '', regex=True)
        data[col] = data[col].str.replace('_', ' ')
        data[col] = data[col].str.replace(',', ' ')

    # Rename the format of the values in those columns to title
    for col in columns_to_clean:
        data[col] = data[col].str.lower()

    return data
