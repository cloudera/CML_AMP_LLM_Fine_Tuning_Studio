
# Fine Tuning Studio - Import Base Models Guide

[User Guide Home](../user_guide.md)

---

## Importing Base Models in Fine Tuning Studio

This guide explains how to import base models from Hugging Face, model registries, or local project files into Fine Tuning Studio to align with fine-tuning tasks. It also covers how to set up a Hugging Face token for downloading models from private repositories during AMP deployment.

---

## Step-by-Step Process:

### 1. Navigate to Import Base Models

- In Fine Tuning Studio, go to the **Import Base Models** tab.
- There are three options:
  - **Import Hugging Face Models**
  - **Import from Model Registry** (coming soon)
  - **Upload from Project Files** (coming soon)

---

### 2. Importing a Hugging Face Model

Before importing models from Hugging Face, especially if you are accessing private models, you need to set your Hugging Face authentication token during AMP deployment. This allows Fine Tuning Studio to access private repositories.

#### Setting Your Hugging Face Token for AMP Deployment:

1. **Generate a Token**:
   - Go to your Hugging Face account: [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
   - Create a new token or copy an existing one.

2. **Set the Token in AMP**:
   - During the deployment process of your AMP environment, ensure that the Hugging Face token is set as an environment variable:
   
   ```bash
   export HUGGINGFACE_TOKEN=your_huggingface_token_here
   ```

   - You can also include this in your deployment configuration if required.

3. **Access Private Models**:
   - Once the token is set, you will be able to import models from both public and private repositories in Hugging Face.

---

### 3. Importing a Hugging Face Model

- **Step 1**: Select the **Import Hugging Face Models** tab.
- **Step 2**: Enter the model name in the input field. The format should be `organization/model`.

#### Example: How to Find the Organization and Model Name:

To find a model on Hugging Face, follow these steps:

1. **Visit Hugging Face Models**:  
   Go to the Hugging Face Models website: [https://huggingface.co/models](https://huggingface.co/models).

2. **Search for the Model**:  
   Use the search bar to find the model you're interested in.

3. **Model Page**:  
   On the model's page, you'll find the full model name, including the organization and model name. For this example, the model name is `NousResearch/Llama-2-7b-hf`.  
   
   - **Organization**: `NousResearch`
   - **Model**: `Llama-2-7b-hf`
   - **Complete Model Name**: `NousResearch/Llama-2-7b-hf`

4. **Copy the Model Name**:  
   Once you have the model name in the correct format, copy it and paste it into Fine Tuning Studio’s **Model Name** field.

#### Example Input:
```
Model Name: NousResearch/Llama-2-7b-hf
```

- **Step 3**: Click the **Import** button to start importing the model.
- **Step 4**: Fine Tuning Studio will load the model, and a success message will confirm the model import.

**Note**: If the model name is incorrect or left empty, an error message will appear.

#### Common Error:
- **Model Name is Empty**:  
  If the model name field is left blank, the system will prompt you to provide the correct model name in the `organization/model` format.

---

### 4. Importing from a Model Registry (Coming Soon)

- **Model Registry Import**: This feature will allow users to import models from a registered CML Model Registry in the workspace. Currently, this feature is coming soon.

---

### 5. Uploading a Model from Project Files (Coming Soon)

- **Upload from Project Files**: This feature will allow users to upload and import foundational models from local project files. Currently, this feature is coming soon.

---

## Troubleshooting:

- **Model Import Issues**: Double-check the `organization/model` format for Hugging Face models.
- **Model Not Found**: Ensure you are using the correct model name when importing from Hugging Face.
