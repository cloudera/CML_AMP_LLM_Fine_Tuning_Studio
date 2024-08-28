import re
import pandas as pd

# Regex pattern to find placeholders in curly braces

def format_template(template: str, row: pd.Series) -> str:
    pattern = r'\{(.*?)\}'
    formatted_string = template
    # Find all placeholders in the template
    placeholders = re.findall(pattern, template)
    
    # Replace each placeholder with the corresponding value from the row
    for placeholder in placeholders:
        if placeholder in row.index:
            formatted_string = re.sub(
                rf'\{{{re.escape(placeholder)}}}',  # Regex pattern to match the placeholder
                str(row[placeholder]),              # Replacement value
                formatted_string
            )
    
    return formatted_string


def extract_eval_column_name(template: str) -> str:
    pattern = r'\{(.*?)\}'
    placeholders = re.findall(pattern, template)
    return placeholders[0] if placeholders else None

