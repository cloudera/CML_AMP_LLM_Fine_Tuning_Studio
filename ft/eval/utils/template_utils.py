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
def fetch_eval_column_name_and_merge_function(dataset_name):
    if dataset_name == "philschmid/sql-create-context-copy":
        return "answer", merge_columns_sql
    if dataset_name == "teknium/GPTeacher-General-Instruct":
        return "response", merge_columns_instruct
    if dataset_name == "s-nlp/paradetox":
        return "en_neutral_comment", merge_columns_toxic
    raise ValueError("Other dataset not supported yet")
