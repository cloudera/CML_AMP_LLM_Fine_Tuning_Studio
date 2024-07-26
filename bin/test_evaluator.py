from ft.eval.mlflow_driver import driver
import pandas as pd
from glob import glob
from ft.eval.eval_job import StartEvaluationRequest


request = StartEvaluationRequest(
    adapter_path='data/adapters/bloom1b1-lora-sql',
    base_model_name='bigscience/bloom-1b1',
    dataset_name='s-nlp/paradetox'
)
response = driver(request)
print(response.metrics)
print(response.csv)
fls = len(glob("*.csv"))
file_name = "results_" + str(fls+1) +".csv"

response.csv.to_csv(file_name,encoding='utf-8')   # just for teting. Should be displayed in the UI