# Fine Tuning Studio - Create Prompts Guide

[User Guide Home](../user_guide.md)

---

## Creating and Customizing Prompts for Fine-Tuning

In Fine Tuning Studio, prompts play a vital role in guiding the model's training. This guide will help you create custom training prompts tailored to your specific datasets and tasks.

### Steps to Create a Prompt:

1. **Prompt Name**: Provide a unique and human-readable name for your prompt. This name will help you identify the prompt in future tasks.
   
   **Example**: `Customer Service Bot Instruction`

2. **Dataset Selection**: Select the dataset that will be used to generate the prompt. You will be able to preview the dataset's column structure to assist in prompt customization.
   
   **Example**: If the dataset has columns like `instruction`, `input`, and `response`, these can be used in your prompt.

3. **Prompt Template**: Define how you want to structure the input that the model will receive. The default template is auto-generated based on your dataset's columns.

    **Default Example**:
    ```
    You are an LLM. Provide a response below.
    
    <Instruction>: {instruction}
    <Input>: {input}
    <Response>:
    ```

    You can modify this template by adding additional instructions or context that aligns with your task.

    **Customized Example**:
    ```
    You are a customer support chatbot. Please respond politely with helpful information based on the provided input.

    <Instruction>: {instruction}
    <Customer Query>: {input}
    <Support Response>:
    ```

4. **Completion Template**: Define the expected output or completion from the model. This is the target the model learns to predict during training.

    **Default Example**:
    ```
    {response}
    ```

    You can customize this as needed to include more context or fields from the dataset.

    **Customized Example**:
    ```
    {response}
    <Source>: {source}
    ```

5. **Generate and Review Examples**: 
    - **Generate Prompt Example**: This feature allows you to preview how the templates will be filled with real data from your dataset. 
    - **Example Training Prompt**: The full input that will be used during training.
    - **Example Prompt and Completion**: Shows a sample prompt and completion based on your template.

6. **Save the Prompt**: Once satisfied, click the **Create Prompt** button to save your work. The prompt will be added to your prompt repository and can be used in training tasks.

---

### Debugging Prompt Creation

While creating prompts, it's important to ensure the templates are compatible with the dataset structure. Errors such as missing or incorrectly formatted fields can lead to issues during training.

Common Errors:

- **Missing Dataset Fields**: Ensure all placeholders in the prompt (e.g., `{instruction}`, `{input}`) match the columns in your dataset.
  
- **Empty Prompt Name**: A prompt name is required. Ensure that it is filled before attempting to save the prompt.

- **Incorrect Dataset Format**: Some datasets may not have the required fields for your template. Adjust the prompt template to reflect the actual dataset structure.

---

### Example Usage Scenarios:

- **Instruction-Based Model Fine-Tuning**:
    - **Prompt Template**: `You are an AI assistant. {instruction} based on the following input: {input}`
    - **Completion Template**: `{response}`

- **Chatbot Fine-Tuning**:
    - **Prompt Template**: `You are a customer service bot. Respond to the following question: {input}`
    - **Completion Template**: `{response}`

---

## Advanced Tips

- Experiment with different prompt and completion templates to see how the model's performance changes.
- Ensure the prompt and completion templates are concise and structured for your specific task.
- Review dataset features carefully to ensure all necessary columns are included in the prompt.
