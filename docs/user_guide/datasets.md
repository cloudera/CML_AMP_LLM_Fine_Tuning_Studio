
# Fine Tuning Studio - Import Datasets Guide

[User Guide Home](../user_guide.md)

---

## Importing Datasets in Fine Tuning Studio

This guide explains how to import datasets from Hugging Face or CSV files into Fine Tuning Studio to train adapters on foundational models.

---

## Step-by-Step Process:

### 1. Navigate to Import Datasets

- In Fine Tuning Studio, go to the **Import Datasets** tab.
- There are two options:  
  - **Import Hugging Face Dataset**  
  - **Import CSV Dataset**

---

### 2. Importing a Hugging Face Dataset

- **Step 1**: Select the **Import Hugging Face Dataset** tab.
- **Step 2**: Enter the dataset name in the input field. The format should be `organization/dataset`.

#### Example: How to Find the Organization and Dataset Name:

To find a dataset on Hugging Face, follow these steps:

1. **Visit Hugging Face Datasets**:  
   Go to the Hugging Face Datasets website: [https://huggingface.co/datasets](https://huggingface.co/datasets).

2. **Search for the Dataset**:  
   Use the search bar to find the dataset you're interested in. For this example, we will import the dataset named `sql-create-context-copy`.

3. **Dataset Page**:  
   On the dataset's page, you'll find the full dataset name, including the organization and dataset name. For this dataset, the name is `philschmid/sql-create-context-copy`.  
   
   - **Organization**: `philschmid`
   - **Dataset**: `sql-create-context-copy`
   - **Complete Dataset Name**: `philschmid/sql-create-context-copy`

4. **Copy the Dataset Name**:  
   Once you have the dataset name in the correct format, copy it and paste it into Fine Tuning Studio’s **Dataset Name** field.

#### Example Input:
```
Dataset Name: philschmid/sql-create-context-copy
```

- **Step 3**: Click the **Import** button to start importing the dataset.
- **Step 4**: Fine Tuning Studio will load the dataset, and a success message will confirm the dataset import.

**Note**: If the dataset name is incorrect or left empty, an error message will appear.

#### Common Error:
- **Dataset Name is Empty**:  
  If the dataset name field is left blank, the system will prompt you to provide the correct dataset name in the `organization/dataset` format.

---

### 3. Importing a CSV Dataset

- **Step 1**: Select the **Import CSV Dataset** tab.
- **Step 2**: Enter the dataset name and the path to the CSV file.

#### Preparing Your CSV File:

To import a CSV dataset, the CSV file should be formatted and placed correctly within the project folder:

1. **CSV Format**:  
   - The first row should contain feature names (column headers).
   - Column headers should not contain whitespace. Instead, use underscores (`_`) or camelCase.
   - The CSV file should contain data that can be cast into a standard dataset format.

2. **Dataset Path**:
   - The path should be relative to the root directory of your AMP project.
   - Place your CSV file in a folder within the project and reference it using a relative path. For example, if your CSV is located at `datasets/my_dataset.csv`, use this as the location.

#### Example:
- **Dataset Name**:  
  ```
  Dataset Name: My Custom Dataset
  ```
- **CSV Location** (relative to the root project directory):
  ```
  CSV Location: datasets/my_dataset.csv
  ```

#### Step-by-Step Instructions:

1. **Place CSV File in Project**:  
   Store your CSV file in a directory within the project, such as `datasets/`.

2. **Enter Dataset Information**:  
   In Fine Tuning Studio, provide the dataset name (e.g., "My Custom Dataset") and the relative path to the CSV file (e.g., `datasets/my_dataset.csv`).

3. **Click Import**:  
   After providing the necessary information, click the **Import** button to load the dataset.

4. **Dataset Loaded**:  
   Once imported, you will see a success message indicating the dataset is ready for use in training adapters.

**Note**: If the dataset path or name is incorrect, the system will display an error message.

---

### Error Handling

#### 1. Hugging Face Dataset Import Errors:
- **Dataset Name is Empty**:  
  If the dataset name is not provided, you will receive an error asking you to enter a valid `organization/dataset` name.
  
- **Dataset Not Found**:  
  If the specified dataset cannot be found on Hugging Face, ensure you are using the correct format and dataset name.

#### 2. CSV Dataset Import Errors:
- **Dataset Name is Empty**:  
  If you do not provide a dataset name, an error will be displayed, asking for the name.

- **Invalid CSV Path**:  
  Ensure the CSV file is in the correct directory and that the path is provided relative to the project root. If the path is incorrect or the file does not exist, an error will occur.

---

### CSV Formatting Guidelines:

- **Feature Names**: Column headers in the CSV should not have spaces. Use underscores (`_`) or camelCase. Example:
  ```
  id,question,answer
  ```

- **Relative Paths**: Always provide the CSV location relative to the project root. Example:
  ```
  datasets/my_custom_dataset.csv
  ```

By following these steps, you can easily import datasets from Hugging Face or upload CSV datasets for use in Fine Tuning Studio.

---

## Troubleshooting:

- **Hugging Face Dataset Import Issues**: Double-check the `organization/dataset` format.
- **CSV Dataset Import Issues**: Ensure the file path is correct and the CSV is formatted properly.

Once imported, these datasets can be used for training adapters on foundational models, fine-tuning for specific tasks, and running evaluations.
