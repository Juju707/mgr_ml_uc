# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 15:34:47 2020

@author: Juju
"""
import os
import pandas as pd
import datetime
import numpy as np
import re
from pathlib import Path
import openpyxl


class Converter:
    
    

    def __init__(self):
        self.dir = os.getcwd()
        self.columns = ['Płeć', 'Wiek']
        self.sub_list = []
        self.final_list = []
        self.separators=["Rozpoznanie choroby:", "Zastosowane procedury:", "Badania laboratoryjne",
                         "Badania laboratoryjne - antybiogram:","Badania inne:", "Podane leki:", "Epikryza:",
                         "Produkty lecznicze wraz z dawkowaniem i wyroby medyczne, w ilościach odpowiadających ilościom na wystawionych receptach:",
                         "Świadczenia:", "Badania diagnostyczne", "Badania inne:" ]

    def create_new_file(self):
        time = datetime.datetime.now()
        pd.ExcelWriter('{}_extraction.xlsx'.format(time.strftime('%Y_%m_%d_%H_%M_%S')), engine='xlsxwriter')

    def get_all_files_in_dir(self):
        files = [file.name for file in Path(self.dir).rglob('*.xls')]
        return files

    def read_file(self, filepath):
        file = pd.read_excel(filepath)
        return file.dropna(how='all', axis=1)

    def get_personal_data(self, df):
        df = df.dropna(how='all', axis=1)  # usuwa NaN, None ale puste Stringi zostawia
        df.columns = (range(7))
        row = df[df[3].str.contains("Płeć|PESEL|Ks.", na=False)]

        data = [row[3].item(), row[4].item(), row[5].item(), row[6].item(), row[4].item()]
        for col in data:
            value = re.search(r":\s(.*)", col)
            data[data.index(col)] = value.group(1)
        data[4] = self.get_age(data[1])
        dict_data = {'Płeć': data[0], 'Wiek': data[4]}
        return dict_data

    def split_df(self, df, idx):
        return np.split(df, [idx], axis=0)

    def has_Numbers(self, inputString):
        return any(char.isdigit() for char in inputString)

    def get_age(self, string):
        sub = string[0:6]
        today = datetime.date.today()
        birth_date = self.get_birth_date(sub)
        time_difference = today - birth_date
        age = int(time_difference.days / 365.25)
        return age

    def get_birth_date(self, string):
        millenium = 19
        year = string[0:2]
        month = string[2:4]
        while int(month) > 12:
            millenium = millenium + 1
            month = int(
                month) - 20
        return datetime.date(int(str(millenium) + str(year)), int(month), int(string[4:]))

    # Rozpoznanie choroby
    def get_diseases_recognition(self, frame):
        header_data = ''
        for index, row in frame.iterrows():
            for idx, item in row.iteritems():
                if isinstance(item, str):
                    header_data = header_data + item + ' '
        return header_data

    # Podane leki
    def get_given_drugs(self, frame, headers_indices):
        value = ''
        for index, row in frame.iterrows():
            for idx, item in row.iteritems():
                if isinstance(item, str) and index != headers_indices[0]:
                    data = item.split("Dawka:")
                    key = data[0].strip()
                    value = value + ';'+ key + " " + data[1].strip()
        return value[1:]

    # Konsultacje
    def get_consultation(self, frame, header, headers_indices, df):
        value = ''
        frame = frame.drop(df.columns[[3]], axis=1)
        for index, row in frame.iterrows():
            for idx, item in row.iteritems():
                if index != headers_indices[0] and isinstance(item, str):
                    value = value + item + ';'
        return value

    # Epikryza
    def get_text(self, frame, header, headers_indices):
        value = ''
        for index, row in frame.iterrows():
            for idx, item in row.iteritems():
                if index != headers_indices[0] and isinstance(item, str):
                    value = value + item + ';'
        return value

    # Badania inne
    def get_other_tests(self, frame, headers_indices):
        isHeader = False
        dic = {}
        for index, row in frame.iterrows():
            for idx, item in row.iteritems():
                if isinstance(item, str) and index != headers_indices[0]:
                    isHeader = not isHeader
                    if isHeader:
                        key = item.strip()
                    else:
                        value = item.strip()
                        dic[key] = value
                        self.columns.append(key.strip())
        return dic

    # na wystawionych receptach
    def get_made_up_by_prescription(self, frame, header, df):
        value = ''
        frame = frame.drop(df.columns[[0, 1, 3, 5, 10, 12]], axis=1)
        for index, row in frame.iterrows():
            for idx, item in row.iteritems():
                if isinstance(item,
                              str) and 'jednostka' not in item and 'Dawkowanie' not in item and 'data' not in item:
                    value = value + item + ','
                elif isinstance(item, str) and 'data' in item:
                    value = value[:-1] + '||' + item[6:] + ','
        return value[2:]

    #  niemożliwy
    def get_impossible(self, sub_string, sub_title, date):
        dic = {}
        sub_idx = sub_string.index('niemożliwy')
        value = ' '.join(sub_string[sub_idx:])
        sub_title = sub_title + ' '.join(sub_string[:sub_idx]).strip()
        key = sub_title + " " + date
        dic[key] = value
        return dic, key

    # dziwne case'y
    def get_special_cases(self, sub_string, sub_title, date):
        dic = {}
        value = sub_string[-1]
        sub_title = sub_title + ' '.join(sub_string[:-1]).strip()
        key = sub_title + " " + date
        dic[key] = value
        return dic,key

    # Badania laboratoryjne
    def get_laboratory_tests(self, frame, headers_indices, patient_data):
        sub_header = ''

        # Przejdź po każdym rzędzie danych
        for index, item in frame.iterrows():
            if isinstance(item.iloc[0], str):
                        item=item.iloc[0].replace('  ', ' ').replace('.', '').strip()
                        if not item.isupper():
                            raw_data = item[11:]  # 11 bo żeby usunąć datę z wyniku badania
                            date = item[0:11]
                            raw_data = raw_data.split(";")
                            for data in raw_data:
                                if data[
                                   0:3] == ' 20':  # sprawdzamy czy to nie przypadkiem kolejne wyniki zaczynające się datą
                                    data = data.strip()
                                    date = data[0:11]
                                    data = data[11:]

                                norm_idx = data.find('(norm')
                                sub_string = data[:norm_idx - 1].split()
                                sub_title = ''
                                if (sub_header.lower() not in data.lower()):
                                    sub_title = sub_header + ' '

                                if len(sub_string) <= 1 or 'KOMENTARZ' in sub_string or 'Komentarz' in sub_string:
                                    continue
                                elif len(sub_string) > 2:
                                    if  ('KWAS' in sub_string and'FOLIOWY' in sub_string) or ('MOCZOWY' in sub_string and 'KWAS' in sub_string) or ('P/CIALA' in sub_title):
                                         value = ' '.join(sub_string[-2:])
                                         sub_title = sub_title + ' '.join(sub_string[:-2]).strip()
                                         key = sub_title + " " + date
                                         patient_data[key] = value
                                         self.columns.append(key[:-11].strip())
                                         
                                    elif ('COV-2' in sub_title):
                                        key=sub_title
                                        value=raw_data[1]
                                        patient_data[key] = value
                                        self.columns.append(key.strip())
                                        
                                    elif ('NUMER' in sub_string and 'BADANIA' in sub_string)or ('QUANTIFERON-' in sub_title) or ('S-METYLOTRANSFERAZA' in sub_title) or ('TIOPURYNY' in sub_title):
                                        break
                                    
                                    elif 'MULTITEST' in sub_string and 'PCR' in sub_string:
                                        key='PATOGENY ODDECHOWE '
                                        value=' '.join(raw_data).strip()
                                        patient_data[key] = value
                                        self.columns.append(key.strip())
                                        break
                                        
                                    
                                    elif 'prep.' in sub_string or 'prep;' in sub_string or 'prep' in sub_string or 'PROTROMBINY 'in sub_title or 'HISTOPATOLOGICZNE'in sub_title or 'oznaczenie' in sub_string or 'nieliczne' in sub_string or 'pojedyncze' in sub_string or'Niemożliwe' in sub_string or 'Uwagi' in sub_string or 'P/CIAŁA' in sub_title or 'HISTOPATOLOGICZNE'in sub_title or ('KAŁ' in sub_title and 'PASOŻYTY' in sub_title) or ('MPV' in sub_string) or ('ROZMAZ - MIKROSKOPOWY'in sub_title) :
                                        keys=[k for k in sub_string if k.isupper()]
                                        vals=[v for v in sub_string if not v.isupper()]
                                        key=sub_title  + ' '.join(keys).strip() + " " + date
                                        self.columns.append(key[:-11].strip())
                                        if len(raw_data)>1 and ('ROZMAZ - MIKROSKOPOWY'in sub_title):
                                            value=' '.join(raw_data).strip()
                                            patient_data[key] = value
                                            break
                                        value=' '.join(vals).strip()
                                        patient_data[key] = value
                                        
                                    elif 'TEST' in sub_string or 'WYNIK' in sub_string or 'toksyna' in sub_string or 'KOLOR' in sub_string or 'PASMA' in sub_string or 'kału' in sub_string or 'Krople' in sub_string or 'Ziarna' in sub_string or 'CIĘŻAR' in sub_string or 'KWAS' in sub_string:
                                        dic, key = self.get_special_cases(sub_string, sub_title, date)
                                        patient_data.update(dic)

                                    elif 'niemożliwy' in sub_string:
                                        dic, key = self.get_impossible(sub_string, sub_title, date)
                                        patient_data.update(dic)
                                        self.columns.append(key[:-11].strip())

                                    else:
                                        value = sub_string[-2] + ' ' + sub_string[-1]  # WYNIK Z JEDNOSTKA
                                        key = sub_title  + ' '.join(sub_string[:-2]).strip() + " " + date
                                        patient_data[key] = value
                                        self.columns.append(key[:-11].strip())
                                else:
                                    value = sub_string[1]
                                    key = sub_string[0] + ' ' + date
                                    key = sub_title + ' ' + key
                                    patient_data[key] = value
                                    self.columns.append(key[:-11].strip())

                                
                        else:
                            sub_header = item
                    
                

    def get_antibiogram(self,df,header,patient_data):
                value = ''
                examination, bacterie = self.split_df(df, 4) #dzieli na dateframe 'Badanie' i 'Nazwa organizmu'
                examination = examination.iloc[1:-1] 
                
                for idx, sub_row in examination.iteritems(): #kolumnami leci pętle po 'Badanie'
                    for i, item in sub_row.iteritems():
                        if isinstance(item, str):
                            value = value + item + ','
                value = value[:-1] + ': '

                for idx, sub_row in bacterie.iterrows(): #pętla po 'Nazwa organizmu'
                    if sub_row.iloc[0] == 'Lp.':
                        value = value[:-1]
                        patient_data[key] = value
                        self.columns.append(key.strip())
                        break

                    bacterium = r'('
                    for i, item in sub_row.iteritems():
                        if isinstance(item, str):
                            bacterium = bacterium + str(item) + ','
                    bacterium = bacterium[:-1] + r')'
                    value = value + bacterium + ';'
                patient_data[header] = value
    
    def get_test_data(self, df):
        patient_data = {}
        df = df.dropna(how='all', axis=1)
        df.drop(df.tail(6).index,inplace=True)
        df.columns = (range(len(df.columns)))
        headers_indices=[]
        for row in df.iterrows():
            if row[1].iloc[0] in self.separators:
                headers_indices.append(row[0])
        split_idx = None

        while len(headers_indices) > 1:  # sprawdza ile jest elementow na liście
            split_idx = headers_indices[1] - headers_indices[0]
            frame, df = self.split_df(df, split_idx)
            frame = frame.dropna(how='all', axis=1)
            header = frame[0][headers_indices[0]]
            frame.drop(frame.head(1).index,inplace=True)

            if header == 'Rozpoznanie choroby:':
                patient_data[header] = self.get_diseases_recognition(frame)
                self.columns.append(header.strip())

            elif header == 'Badania laboratoryjne':
                self.get_laboratory_tests(frame, headers_indices, patient_data)
            
            elif header == 'Badania laboratoryjne - antybiogram:':
                self.get_antibiogram(frame, header, patient_data)
                self.columns.append(header.strip())

            elif header == 'Podane leki:':
                patient_data[header] = self.get_given_drugs(frame, headers_indices)
                self.columns.append(header.strip())
                
            elif header == 'Badania inne:':
                patient_data[header] = self.get_text(frame, header, headers_indices)
                self.columns.append(header.strip())

            elif header == 'Epikryza:' or header == 'Zastosowane procedury:' or header == 'Świadczenia:':
                patient_data[header] = self.get_text(frame, header, headers_indices)
                self.columns.append(header.strip())

            else :
                new_header='Nowe: ' + header
                patient_data[new_header] = self.get_text(frame, header, headers_indices)
                self.columns.append(new_header.strip())
                
            
            headers_indices.pop(0)

        return patient_data

    def add_file_data(self, personal_data, test_data):
        return pd.concat([personal_data, test_data], axis=1)

if __name__ == "__main__":
    conv = Converter()
    lista = conv.get_all_files_in_dir()
    for file in lista:
        file0 = conv.read_file(file)
        psl2 = conv.split_df(file0, 5)
        person = conv.get_personal_data(psl2[0])
        tests = conv.get_test_data(psl2[1]) 
        person.update(tests)
        data = person.copy()
        conv.final_list.append(data)

    conv.columns = list(set(conv.columns))
    df = pd.DataFrame(columns=conv.columns)
    for dictionary in conv.final_list:
        row = {k: '-' for (k, v) in df.items()}
        for column in df:
            val = ''
            for key, value in dictionary.items():
                if column.strip() in key:
                    if column in ['Płeć', 'Wiek']:
                        val = str(value)
                        break
                    else:
                        key.strip()
                        a=column.strip().replace("  "," ")
                        b=key.strip().replace("  "," ")
                        if len(key)>11 and key[-11:-9]=='20':
                            b=b[0:-11].strip()
                        c=(a==b)
                        if c:
                            val = val + str(value) + ';'
                            
            if len(val) > 0:
                if val[0] == ';':
                    val = val[1:]
                    row[column] = val

                else:  
                    row[column] = val

        df = df.append(row, ignore_index=True)
    df = df.reindex(sorted(df.columns), axis=1)
    df.to_excel(r'Eksport.xlsx', index=False)
