
# Fine Tuning Studio - Local Adapter Comparison Guide

[User Guide Home](../user_guide.md)

---

## Comparing Fine-Tuned Adapters with Base Models

This guide outlines how to configure, compare, and evaluate fine-tuned adapters against base models using the Fine Tuning Studio interface.

### Step-by-Step Process:

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
   Adapter: LoRA-Adapter-for-Llama-2-7b-hf
   ```

3. **Prompt and Completion Template**:
   - Prompts define the input to the model, while completion templates define the expected output.
   - The system will generate a prompt example based on real dataset values, and display both the input prompt and the expected completion.

   **Prompt Template Example**:
   ```
   You are an AI assistant. Answer the following question:
   
   Question: {question}
   ```

   **Completion Template Example**:
   ```
   Answer: {answer}
   ```

4. **Random Prompt Generation**:
   - You can generate a random prompt using the selected template and the dataset associated with the model.

5. **Evaluate and Compare**:
   - After configuring the model, prompt, and adapters, click **Generate** to run the model and its adapters.
   - The base model output is shown alongside the output from each selected adapter for easy comparison.

### Error Handling:

- **Model Not Selected**: 
   - Ensure a base model is selected before running the comparison.

- **No Adapters Found**: 
   - If no adapters are found, you may need to fine-tune the base model first.

---

### Example Scenarios:

#### Instruction-Based Adapter Fine-Tuning:

1. **Adapter Prompt Template**:
   ```
   You are a helpful assistant. Please answer the following:
   
   Question: {question}
   ```

2. **Base Model Completion**:
   ```
   Answer: The assistant helps you with the following information...
   ```

3. **Adapter Completion**:
   ```
   Answer: The fine-tuned adapter provides more detailed responses...
   ```

By comparing the base model and fine-tuned adapter, you can evaluate how the adapter has improved performance in specific tasks.

