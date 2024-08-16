from sys import argv
from pandas import read_xml, DataFrame, to_datetime
from datetime import datetime
from numpy import array_split
from re import split

args = argv[1:]
df = read_xml(args[0],encoding='utf-8')
df.CloseDate = to_datetime(df.CloseDate.astype(str),errors='coerce',format='%m%d%Y.0')
df=df.drop(df[(df['CloseDate']<datetime.now())].index)


#print('GRANT',df)
df.to_csv(f'{args[0][:-4]}.csv',index=False)

#cdf=DataFrame(columns=df.columns)
#for idx in df.index:# <- increases hits with GFORWARD which i am testing downweighting
    #print(idx,len(cdf))
    #description=df.loc[idx].Description
    #if isinstance(description,float):
        #continue
    #for c in split(r'(.*)',description):
        #if c.strip():
            #row=DataFrame(df.loc[idx])
            #row['Description']=c
            #cdf.loc[len(cdf)]=row.iloc[0]
#print(cdf)
#[print('\n',_,'\n') for _ in df.iloc[:100].Description]
#for idx, chunk in enumerate(array_split(df,10*len(df)/int(args[1]))):
    #i=str(idx).zfill(3)
    #chunk.to_csv(f'{args[0][:-4]}_S{i}',index=False)
