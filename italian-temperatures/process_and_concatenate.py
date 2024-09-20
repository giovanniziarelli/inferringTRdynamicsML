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

def find_csv_files(root_dir):
    csv_files = []
    for root, dirs, files in os.walk(root_dir):
        dirs.sort()
        #print(files)
        #files = ['Gennaio.csv', 'Febbraio.csv', 'Marzo.csv', 'Aprile.csv', 'Maggio.csv', 'Giugno.csv', 'Luglio.csv', 'Agosto.csv', 'Settembre.csv', 'Ottobre.csv', 'Novembre.csv', 'Dicembre.csv']  # replace with your actual file paths
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    return csv_files

def concatenate_csv_files(file_paths, columns_to_extract):
    dataframes = []
    for file_path in file_paths:
        print(file_path)
        df = read_custom_csv(file_path)
        #print(df)
        if len(df) == 0:
            pass
        # Extract only the desired columns
        else:
            df = df[columns_to_extract]
            df = df.dropna()
            print(len(df.columns))
            df['DATA'] = pd.to_datetime(df['DATA'], format = "%d/%m/%Y")
            dataframes.append(df)
    
    # Concatenate all DataFrames
    concatenated_df = pd.concat(dataframes, ignore_index=True)
    print(concatenated_df)
    concatenated_df.sort_values(by='DATA', inplace=True)
    return concatenated_df


# Usage
# List of file paths to be concatenated
#file_paths = ['Gennaio.csv', 'Febbraio.csv', 'Marzo.csv', 'Aprile.csv', 'Maggio.csv', 'Giugno.csv', 'Luglio.csv', 'Agosto.csv', 'Settembre.csv', 'Ottobre.csv', 'Novembre.csv', 'Dicembre.csv']  # replace with your actual file paths
#concatenated_df = concatenate_csv_files(file_paths)[columns_to_extract]
#print(concatenated_df)

# Usage
# Root directory containing folders with CSV files
root_dirs = ['Torino', 'Genova', 'Milano', 'Trento', 'Venezia', 'Trieste', 'Bologna', 'Firenze', 'Ancona', 'Roma', 'Bari', 'Napoli', 'Catanzaro', 'Palermo', 'Cagliari']
for i in range(len(root_dirs)):
    root_dir = root_dirs[i] + '/'  # replace with your actual root directory

    # Find all CSV files in the root directory and subdirectories
    file_paths = find_csv_files(root_dir)

    # List of columns to extract
    #columns_to_extract = ['Column1', 'Column2', 'Column3']  # replace with your actual column names
    columns_to_extract = ['DATA', 'TMEDIA','TMIN','TMAX', 'UMID']

    # Concatenate the CSV files and extract the specific columns
    concatenated_df = concatenate_csv_files(file_paths, columns_to_extract)
    print(concatenated_df)

    # Optionally, save the concatenated DataFrame to a new CSV file
    concatenated_df.to_csv(root_dirs[i] +'_concatenated.csv', index=False, sep=';', quotechar='"', quoting=1)

    # Optionally, save the concatenated DataFrame to a new CSV file
    #concatenated_df.to_csv('concatenated_file.csv', index=False, sep=';', quotechar='"', quoting=1)

