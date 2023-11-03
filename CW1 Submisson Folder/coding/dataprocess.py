#!/usr/bin/env python
# coding: utf-8

# In[21]:


import pandas as pd
from tqdm import tqdm

# Read Excel file
file_path = './X_tr.xlsx'
sheet_names = pd.ExcelFile(file_path).sheet_names

# Read label
df_label = pd.read_excel('./Y_tr.xlsx')
# del df_label['Unnamed: 0']
# Iterate through each row
for row_index in range(37):
    # Initialize an empty DataFrame to store the spliced data
    label = df_label[row_index+1] #
    concatenated_df = pd.DataFrame()
    # concatenated_df = pd.concat([label,concatenated_df],ignore_index = True)

    # Traverse each sheet
    for sheet_name in tqdm(sheet_names):
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        
        # Select the data of the current row and transpose
        row_data = df.iloc[row_index].values.reshape(1, -1)
        
        # Splice the current row data into concatenated_df
        concatenated_df = pd.concat([concatenated_df, pd.DataFrame(row_data)], ignore_index=True)

    concatenated_df.columns = ['position'] + ['F{}'.format(i) for i in range(1, 12)]
    concatenated_df['label'] = label
    # Save the spliced data as a CSV file with the file name "{row_index + 1}.csv"
    concatenated_df.to_csv(f'./data/{row_index + 1}.csv', index=False)


# In[ ]:




