'''
Converts xml to csv

Reads argument from command line
'''
from sys import argv
from datetime import datetime
from pandas import read_xml, to_datetime

args = argv[1:]
df = read_xml(args[0], encoding='utf-8')
df.CloseDate = to_datetime(df.CloseDate.astype(str),
                           errors='coerce',
                           format='%m%d%Y.0')
df = df.drop(df[(df['CloseDate'] < datetime.now())].index)
df.to_csv(f'{args[0][:-4]}.csv', index=False)
