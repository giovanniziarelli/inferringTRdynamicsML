import urllib.request
import os
import csv
opener = urllib.request.URLopener()
opener.addheader('User-Agent', 'whatever')

cities = ['Torino', 'Genova', 'Milano', 'Trento', 'Venezia', 'Trieste', 'Bologna', 'Firenze', 'Ancona', 'Roma', 'Bari', 'Napoli', 'Catanzaro', 'Palermo', 'Cagliari']
years  = ['2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020']
months = ['Gennaio', 'Febbraio', 'Marzo', 'Aprile', 'Maggio', 'Giugno', 'Luglio', 'Agosto', 'Settembre', 'Ottobre', 'Novembre', 'Dicembre']

# Function to modify the first row of a CSV file
def modify_first_row(csv_file, old_w, new_w):
    with open(csv_file, 'r', newline='') as file:
        reader = csv.reader(file)
        rows = list(reader)
        if len(rows) > 0:
            # Modify the first row here
            # For example, you can capitalize all column headers
            rows[0] = [cella.replace(old_w, new_w) for cella in rows[0]]

    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows)

old_w_s = ['TMAX °C','TMEDIA °C','TMIN °C', 'UMIDITA %'] 
new_w_s = ['TMAX', 'TMEDIA', 'TMIN', 'UMID']

for i in range(len(cities)):
    folder_name = cities[i]
    os.makedirs(folder_name, exist_ok = True)
    for j in range(len(years)):
        y_folder_name = folder_name + '/' +years[j]
        os.makedirs(y_folder_name, exist_ok = True)
        for k in range(len(months)):
            url_name  = 'https://www.ilmeteo.it/portale/archivio-meteo/' + cities[i] + '/' + years[j] + '/' + months[k] + '?format=csv'
            file_name = months[k] + '.csv'
            filename, headers = opener.retrieve(url_name,y_folder_name + '/'+ file_name)
        for k in range(len(old_w_s)):
            old_w = old_w_s[k]
            new_w = new_w_s[k]
            # Iterate over files in the directory
            for filename in os.listdir(y_folder_name):
                if filename.endswith('.csv'):
                    csv_file = os.path.join(y_folder_name, filename)
                    modify_first_row(csv_file, old_w, new_w)

