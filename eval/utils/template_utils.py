def merge_columns_instruct(example):
    if example["input"]:
        prediction_format = """<Instruction>: %s
<Input>: %s
<Response>:"""
        formatted_text = prediction_format % (example["instruction"], example["input"])
    else:
        prediction_format = """<Instruction>: %s
<Response>:"""
        formatted_text = prediction_format % (example["instruction"])
    return formatted_text


def merge_columns_sql(example):
    prediction_format = """<TABLE>: %s
<QUESTION>: %s
<SQL>:"""
    formatted_text = prediction_format % (example["context"], example["question"])
    return formatted_text


def merge_columns_toxic(example):
    prediction_format = """<Toxic>: %s
<Neutral>:"""
    formatted_text = prediction_format % (example["en_toxic_comment"])
    return formatted_text


# Hardcoding this for demo
def fetch_eval_column_name_and_merge_function(dataset_id):
    if dataset_id == "6e8c9e5a-3168-44bb-b6ee-72f59132f77e":
        return "answer", merge_columns_sql
    if dataset_id == "a674cd4a-cbcf-490b-ba21-8db2ef689edd":
        return "response", merge_columns_instruct
    if dataset_id == "24e91769-1d0e-4eb8-8ac2-50032598d66f":
        return "en_neutral_comment", merge_columns_toxic
    raise ValueError("Other dataset not supported yet")
