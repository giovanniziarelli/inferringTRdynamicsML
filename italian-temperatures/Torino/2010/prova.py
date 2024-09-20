import pandas as pd

def read_custom_csv(file_path):
    with open(file_path, 'r') as file:
        # Read the header line
        header_line = file.readline().strip()
        # Process the header line
        headers = header_line.split(';')
        
        # Initialize an empty list to store the processed rows
        rows = []
        
        # Read and process each subsequent line
        for line in file:
            # Strip newline characters and split the first two fields by ';'
            fields = line.strip().split(';', 2)
            if len(fields) > 2:
                # For the remaining fields, split by '";"'
                rest_fields = fields[2].split('";"')
                # Remove extra quotes if present
                rest_fields = [field.strip('"') for field in rest_fields]
                # Combine the first two fields with the rest
                fields = fields[:2] + rest_fields
            rows.append(fields)
        
        # Convert the list of rows into a DataFrame
        df = pd.DataFrame(rows, columns=headers)
        
        return df

# Usage
file_path = 'Gennaio.csv'
df0 = pd.read_csv(file_path)
print(df0)
df = read_custom_csv(file_path)
print(df)

