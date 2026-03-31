# Project Proposal: Cardiovascular Disease Prognosis

## Group Members
- **[Filipe Viseu - 119192](https://github.com/FilipeNV1)**
- **[Duarte Branco - 119253](https://github.com/duartebranco)**
- **[Samuel Vinhas - 119405](https://github.com/samuelvinhas)**

## Context and Motivation
"Cardiovascular diseases are the leading cause of death globally. While early detection significantly improves survival rates, many individuals do not have immediate access to complete clinical blood panels."

Our objective is to develop a ML based Decision Support System that predicts a user's probability of having cardiovascular disease based on accessible metrics. Our primary motivation is to build an adaptive tool: acknowledging that a general user might know their height, weight, and age, but might not know their exact glucose or cholesterol levels. Instead of failing, our system will adapt to missing inputs, providing a baseline risk probability that updates as the user inputs more clinical data.

## Dataset description
- **Origin**: We will utilize the "Cardiovascular Disease dataset" (uploaded by S. Ulianova on Kaggle), which contains 70,000 clinical records of patients, incorporating 11 objective, subjective, and examination features (e.g., age, height, weight, blood pressure, cholesterol, glucose, smoking habits).
- **Link**: [Cardiovascular Disease dataset on Kaggle](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)
- **Limitations and Potential Bias**: 65% of the records are from male patients, and 35% from female patients, which may introduce gender bias. There are also impossible physiological values (e.g., negative blood pressures, or systolic pressures recorded as 14000 instead of 140 due to human typing errors) which will require data cleaning and preprocessing.

## Type of ML Problem
This is a Binary Classification problem.
However, rather than just outputting a rigid 0 (Healthy) or 1 (Heart Disease) classification, we will make our algorithms output a Risk Probability Percentage. This aligns with medical best practices, giving the user a gradient of their risk rather than a definitive, simple diagnosis.

## Initial Risks and Assumptions
- **Risk 1: Handling Missing Inputs at Inference**: 
    - The core feature of our application is allowing users to leave fields (like glucose) as "unknown".
    - Assumption/Mitigation: We will implement a dynamic strategy that can predict even with missing inputs which can lead us to false positives. To mitigate this, we will train multiple models on different subsets of features (e.g., one model trained on all features, another trained only on non-glucose features, etc.) and use the appropriate model based on the user's input.
- **Risk 2: False Negatives**:
    - In our context (medical context), a False Negative (telling a sick patient they are healthy) is vastly more dangerous than a False Positive.
    - Assumption/Mitigation: We will address this by prioritizing recall during model training and evaluation, ensuring that we minimize false negatives even if it means accepting a higher false positive rate. Additionally, we will provide clear disclaimers about the tool being a supplementary aid rather than a definitive diagnostic tool.