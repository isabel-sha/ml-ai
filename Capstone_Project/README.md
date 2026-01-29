# **ServiceNow Incident Prediction Model - Streamlit App**

This repository contains a machine learning model deployed using Streamlit that predicts the weekly ServiceNow incident counts for both opened and closed incidents. The model allows users to upload a dataset and view predictions through a simple and user-friendly interface.

**Deployed App:** [ServiceNow Incident Prediction](https://5hgcrhgfm3whv4d74kyg8g.streamlit.app/)

### **How to Use the App**

1. **Upload Dataset**: Upload your incident CSV dataset containing the required "opened" and "closed" columns.
   
2. **Click "Report" Tab**: After uploading the dataset, click the **"Report"** tab to view predictions for both opened and closed incident counts.

## **Setup Instructions**

### Running Locally

To run this project locally, follow these steps:

```bash
# Clone the repository
git clone https://github.com/isabel-sha/ml-ai.git

# Navigate to the project directory
cd ml-ai/Capstone_Project

# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment (macOS/Linux)
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Run the Streamlit app
streamlit run weekly_predict_model.py
