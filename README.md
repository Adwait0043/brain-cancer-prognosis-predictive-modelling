# Enhancing Brain Cancer Prognosis: Predictive Modelling for Progression and Survival Rates

## **Overview** 
This project focuses on improving brain cancer prognosis by predicting tumour progression and survival rates using a combination of image data and quantitative patient features. 
It employs machine learning algorithms, including **Random Forest** and **Convolutional Neural Networks (CNN)**, 
to build predictive models for tumour classification and overall survival, achieving high accuracy for clinical application. 

## Technologies Used

1. Programming Language - Python
2. Machine Learning Models:
    -- Random Forest 
    -- Convolutional Neural Network 
    -- Simple Neural Network 
3. Libraries and Tools:
    -- Scikit-learn
    -- TensorFlow
    -- Keras
    -- Pandas
    -- NumPy


## Data Flow

1. Data Collection: Extracted patient data and medical records from the Kaggle dataset for brain tumour analysis, including age, gender, tumour characteristics, and other clinical details.
2. Data Processing: Conducted data cleaning, feature engineering, and normalization using Pandas and NumPy for preparing the data for model training.
3. Modeling: Implemented and trained machine learning models (Random Forest, CNN, Simple Neural Network) to predict brain tumour classifications and patient survival rates.
4. Evaluation: Assessed model performance based on accuracy, precision, recall, and F1-score.
5. Visualization: Used Matplotlib and Seaborn for data visualization and exploratory data analysis (EDA).

## Detailed Project Report (Link):
1. [Report.pdf](https://github.com/Adwait0043/brain-cancer-prognosis-predictive-modelling/blob/main/CS5500_2357954.pdf)

## Datasets Used (Format : csv)

### Overview : 
This research used a secondary data collection method. The dataset used for this study is from Kaggle and named 
“Brain Tumor Data” where specific information regarding the patient is obtained such as age, gender, history, and the nature of the tumour. 
The dataset is deliberately downloaded and analyzed to verify the completeness and relevance of the data.

1. [Brain Tumor.csv](https://github.com/Adwait0043/brain-cancer-prognosis-predictive-modelling/blob/main/Brain%20Tumor.csv)
2. [bt3_dataset.csv](https://github.com/Adwait0043/brain-cancer-prognosis-predictive-modelling/blob/main/bt_dataset_t3.csv)

## Scripts for project 
1. [Jupyter Notebook File](https://github.com/Adwait0043/brain-cancer-prognosis-predictive-modelling/blob/main/CS5500_2357954_Code.ipynb)


## Motivation

This project aims to leverage machine learning for improving the prognosis of brain cancer patients by developing 
models capable of providing accurate and personalized predictions of tumour progression and survival. 
It seeks to advance clinical decision-making and patient outcomes through innovative data-driven approaches.

**Real-World Use Case**

**Healthcare Providers:** Predict patient outcomes and tailor personalized treatment strategies for brain cancer patients.

**Medical Researchers:** Analyze tumour progression data to discover new biomarkers and therapeutic targets.

**Clinicians:** Utilize predictive models to make informed treatment decisions based on genomic, clinical, and imaging data.



## Challenges Faced

**1. Data Quality Issues**

**Solution:**
Addressed issues related to missing values and inconsistencies in the dataset to improve model accuracy.

**2. Model Interpretability**

**Solution:** 
Enhanced the transparency of the Random Forest model to provide insights into the factors driving tumour predictions.

**3. Scalability**

**Solution:**
Used CNN to handle large volumes of image data and ensure accurate tumour classification.
