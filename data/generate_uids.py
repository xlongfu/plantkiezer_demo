import pandas as pd 
import uuid

df = pd.read_csv('data/texas_plant_list_cleaned.csv')

seed = 42

df['uid'] = df.index.map(lambda i: str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{seed}-{i}")))

df.insert(0, 'uid', df.pop('uid'))

df.to_csv('data/texas_plant_list_final.csv', index=False)
