import pandas as pd
import sys


filename = sys.argv[1]
classes = sys.argv[2]
classes = classes.strip(']').strip('[').split(',')



df = pd.read_csv(filename,  delimiter = ',', names = ['_id', 'user', 'location', 'image', 'tag', 'created', '__v'])

def clean_tag(row):
    tag = row['tag']
    for c in classes:
        if c in tag:
            row['tag'] = c
            return row
    return ""

def fix(df, filename):
    with open(filename, "w") as fw:
        fw.write("_id,user,location,image,tag,created,__v" + "\n")
        i = 0
        for index, row in df.iterrows():
            if i != 0:
                row = clean_tag(row)
                line = ''
                if len(row) != 0:
                    for col in row.values:
                        line += str(col)
                        line += ','
                    fw.write(line + "\n")
            i += 1


fix(df, filename +'_fixed')
