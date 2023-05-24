import pandas as pd
import model 
df = pd.read_csv(r'C:\Users\Ashut\Downloads\CS295P - New Model\CS295P - New Model\PK_Ferguson Data.xlsx - Prof Data annotation.csv')
df = model.call_everything(df)
df.to_csv('output.csv', index=False)