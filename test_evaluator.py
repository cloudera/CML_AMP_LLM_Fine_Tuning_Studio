from eval.mlflow_driver import driver
import pandas as pd
from glob import glob

results, results_df = driver('s-nlp/paradetox', 'bigscience/bloom-1b1',
                             'data/adapters/bloom1b1-lora-sql', device="gpu")
print(results)
print(results_df)
fls = len(glob("*.csv"))
file_name = "results_" + str(fls+1) +".csv"

results_df.to_csv(file_name,encoding='utf-8')   # just for teting. Should be displayed in the UI