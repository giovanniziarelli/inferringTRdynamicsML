import re

mesi = ['01','02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
giorni_per_mese = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
def ngiorni_a_data(n_days):
    #n_days è il numero di giorni a partire dal 01/01/2020
    sum = 0;
    curr_m = 0;
    n_anni = 0
    while sum + giorni_per_mese[curr_m] < n_days:
        sum += giorni_per_mese[curr_m]
        curr_m += 1
        if curr_m >= 12:
            curr_m = curr_m % 12
            n_anni += 1
        if n_anni >= 1:
            giorni_per_mese[1] = 28
    mese = str(mesi[curr_m])
    giorno = str(n_days - sum)
    anno = str(2020 + n_anni)
    data = giorno + '/' + mese + '/' + anno
    return data

def data_a_ngiorni(data):
    #restituisce il numero di giorni a partire dal 01/01/2020
    d, m, y = (int(s) for s in (re.findall(r'\b\d+\b', data)))
    giorni_prec = 0
    for curr_m in range(m)[1:]:
        giorni_prec = giorni_prec + 31 - 3 * (curr_m == 2) + (-1) * (
                curr_m == 4 or curr_m == 6 or curr_m == 9 or curr_m == 11)
    giorni_prec = giorni_prec + 1 * ((y == 2020 and (m >= 2 or m==2 and d == 29)) or y>2020 ) + 365 * (y == 2021) 
    giorni_prec = giorni_prec + d;
    return giorni_prec