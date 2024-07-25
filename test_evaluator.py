from eval.mlflow_driver import driver

results, results_df = driver('s-nlp/paradetox', 'bigscience/bloom-1b1',
                             'data/adapters/bloom1b1-lora-sql', device="gpu")
print(results)
print(results_df)