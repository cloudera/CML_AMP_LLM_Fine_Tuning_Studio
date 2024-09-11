# Running MLFlow Evaluations

[User Guide Home](index.md)

## Evaluating a fine tuned adapter using MLFlow

This guide shows how to configure and evaluate fine-tuned adapters on Evaluation dataset using Fine tuning studio interface.

### Step-by-Step Process

1. **Base Model Selection**: 
   - Choose a base model for comparison from the list of available models.
   
   **Example**:
   ```
   Base Model: NousResearch/Llama-2-7b-hf
   ```

2. **Adapter Selection**:
   - After selecting the base model, you can choose one or more fine-tuned adapters associated with the model.
   - Adapters are filtered based on a common `prompt_id` to ensure compatibility.

   **Example**:
   ```
   Adapter: llama-2-sql
   ```

3. **Dataset Selection**: 
   - Choose a dataset to be used for evaluation from the list of available datasets.
   
   **Example**:
   ```
   Base Model: philschmid/sql-create-context-copy

   ```
4. **Prompt and Completion Template Selection**:
   - Prompts define the input to the model, while completion templates define the expected output.

   **Prompt Template Example**:
   ```
    <TABLE>: {context}
    <QUESTION>: {question}
    <SQL>:
   ```

   **Completion Template Example**:
   ```
   {answer}
   ```
5. **Dataset Fraction Selection**:
   - Percentage of the dataset to be used for evaluation. For quick results, use a smaller number such as 0.5

6. **Extra Columns Selection**:
   - Choose extra features/columns that you want to be present in the evaluation results CSV. Leave it empty if you don't want any aditional columns other than input, expected_output, output and columns for metrics.

7. **Advanced Option Selection**:
   - Choose advance training parameters such as **cpu** **memory** and **gpu-type** along with **generational config**.


8. **Start MLFlow Evaluation Job**:
   - After configuring the above options, click **Start MLFlow Evaluation Job** to run the evaluation.
   - Toggle to the view MLFLow Runs page to see the job status and view the evaluation aggregated metrics as well as row wise metrics.

### Error Handling:

- **Model Not Selected**: 
   - Ensure a base model is selected before running the comparison.

- **No Adapters Found**: 
   - If no adapters are found, you may need to fine-tune the base model first.

- **No Prompt Found**
   - If no prompts are found, you may need to create a new prompt for the dataset selected
