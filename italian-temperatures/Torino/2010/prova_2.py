import pandas as pd
import os

def read_custom_csv(file_path):
    with open(file_path, 'r') as file:
        # Read the header line
        header_line = file.readline().strip()
        headers = header_line.split(';')
        
        rows = []
        
        for line in file:
            fields = line.strip().split(';', 2)
            if len(fields) > 2:
                rest_fields = fields[2].split('";"')
                rest_fields = [field.strip('"') for field in rest_fields]
                fields = fields[:2] + rest_fields
            rows.append(fields)
        
        df = pd.DataFrame(rows, columns=headers)
        
        return df

def concatenate_csv_files(file_paths):
    dataframes = []
    for file_path in file_paths:
        df = read_custom_csv(file_path)
        dataframes.append(df)
    
    # Concatenate all DataFrames
    concatenated_df = pd.concat(dataframes, ignore_index=True)
    
    return concatenated_df

# Usage
# List of file paths to be concatenated
file_paths = ['Gennaio.csv', 'Febbraio.csv', 'Marzo.csv', 'Aprile.csv', 'Maggio.csv', 'Giugno.csv', 'Luglio.csv', 'Agosto.csv', 'Settembre.csv', 'Ottobre.csv', 'Novembre.csv', 'Dicembre.csv']  # replace with your actual file paths
columns_to_extract = ['DATA', 'TMEDIA','TMIN','TMAX']
concatenated_df = concatenate_csv_files(file_paths)[columns_to_extract]
print(concatenated_df)

# Optionally, save the concatenated DataFrame to a new CSV file
concatenated_df.to_csv('concatenated_file.csv', index=False, sep=';', quotechar='"', quoting=1)

