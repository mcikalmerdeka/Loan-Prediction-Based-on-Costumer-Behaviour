# Loan Prediction Based on Customer Behavior

![Project Header](https://raw.githubusercontent.com/mcikalmerdeka/Loan-Prediction-Based-on-Costumer-Behaviour/refs/heads/main/Assets/Project%20Header.jpg)

**This is the repository for the final project of the Rakamin Data Science Bootcamp Batch 39 that I attended. We are required to go through the end-to-end process of a data science project analyzing factors influencing the default (failure to pay) of a lending company. The original progress and result of each stage can be seen in the "Original Indonesian Version" folder, while what is displayed here is the result of the conversion from Indonesian to English, accompanied by further improvements made after the completion of the final project.**

# Members of Group 3 (**Dackers**)

1. Muhammad Cikal Merdeka (Leader)
2. Maulana Rifan Haditama
3. Maulana Ibrahim
4. Maria Meidiana Siahaan
5. Revita Rahmadini
6. Nugraha Eddy Wijayanto
7. Mochamad Ali Mustofa

After the new update for further improvement the repository is organized into 4 main folder:

- The `analysis` folder contains the main ipynb files that i used for the EDA, data preprocessing, train and save ML model, and answering business questions and formulating recommendations.
- The `data` folder contains raw and processed data used in the project.
- The `models` folder contains joblib files of trained model.
- The `scripts` folder contains the scripts necessary to run the streamlit application.

## 📌 Problem Statement

A lending company needs to verify loan applications from its prospective borrowers (customers). The manual verification procedure conducted by the company has resulted in numerous inaccurate assessments, leading to loans being granted to borrowers who ultimately fail to repay (default). This has proven detrimental to the company as it incurs financial losses without any return. Additionally, the manual verification process also consumes a considerable amount of time which is not efficient.

Building on this issue, the company aims to develop an automated system that can predict the creditworthiness of prospective borrowers in the future based on past borrower data to reduce the selection of potential customers with high default risks.

## 📌 Role

As a Data Scientist Team, our role involves conducting analysis to gain business insights and designing models that have an impact on business development.

## 📌 Goals

- Enhanced credit risk evaluation: Improve the assessment of credit risk by implementing machine learning models. (**MAIN**)

- Increased efficiency: Streamline the credit risk assessment process for greater effectiveness and reduced processing time. (**SECONDARY**)

## 📌 Objectives

The ultimate goal of this project is to create a machine learning model that can:

- Predict credit risk with a high recall rate. The objective is to reduce the risk of losses due to inaccurate risk assessments.

- Provide automated credit risk assessment decisions in a short amount of time.

## 📌 Business Metrics

- Default Rate (%) : The percentage of prospective customers who fail to repay a loan (default) divided by the total number of customers. (**MAIN**)

- Approval Time : The time taken to process loan applications. (**SECONDARY**)

## 📌The streamlit app from this project that can be tried with these commands:

- Clone repo
```python
git clone https://github.com/mcikalmerdeka/Loan-Prediction-Based-on-Costumer-Behaviour.git
```

- Navigate to streamlit file
```python
cd scripts
streamlit run app.py
```

Just follow the instructions inside the streamlit app and try it out.