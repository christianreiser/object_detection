import pandas as pd
import numpy as np
import csv

df=pd.read_csv('annotations_all.txt',low_memory=False)
current_row = -1
fname_row = 0 

#go through all rows
while current_row < 108641: 
    current_row = current_row + 1

    # if row describes filename
    if df.at[current_row,'A'][0:2] == 'IM':
        
        # delete old filename row
        if current_row != 0:
            df = df.drop([int(fname_row)])
            # current_row = current_row - 1

        # get row index of new filename row
        fname_row = current_row

    # if row destrices object
    elif df.at[current_row,'A'][0:2] != 'IM':
        df.at[current_row,'H'] = df.at[current_row,'B'] + df.at[current_row,'D']
        df.at[current_row,'G'] = float(df.at[current_row,'A']) + df.at[current_row,'C']
        df.at[current_row,'F'] = df.at[current_row,'B']
        
        # handle classes
        if df.at[current_row,'E'] == 'b': #if bird
            df.at[current_row,'D'] = 1 #1 means bird
        elif df.at[current_row,'E'] == 'n': #if non bird 
            df.at[current_row,'D'] = 2 #2 means non bird
        elif df.at[current_row,'E'] == 'u': #if undefined 
            df.at[current_row,'D'] = 3 #3 means undefined
        else:
            print('WARNING! class not covered',df.at[current_row,'E'])

        df.at[current_row,'E'] = df.at[current_row,'A']
        df.at[current_row,'C'] = 3744 
        df.at[current_row,'B'] = 5616 
        df.at[current_row,'A'] = df.at[fname_row,'A']
    else:
        print('WARNING! A not covered')

df.to_csv('annotations_all.txt_edited')

print('done.\n please delete inex column and last filename row at end of csv file')
