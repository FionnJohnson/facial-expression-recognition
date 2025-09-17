# **Facial Expression Recognition (FER) project**


This project investigates how well different facial expressions are classified using 2d landmark data. The four expressions 
tested were happy, sadness, disgusted and surprised, and the data included multiple time frames as subjects moved from a neutral 
face to the expression. 

The project combines machine learning and statistical shape analysis to explore how well geometric features (landmarks) can
predict expressions. Multiple experiments were carried out using different numbers of time points. Furthermore, the geometric 
separability of expression pairs was also investigated.

<br/>

## Methods:

Data Preprocessing  
- Generalized Procrustes Alignment (GPA) to standardize facial shapes.  
- Landmark matrices converted into vectorized form.

Dimensionality Reduction  
- Principal Component Analysis (PCA) to reduce thousands of coordinates into the top components capturing most variation.

Classification:  
- Logistic Regression  
- k-Nearest Neighbors (KNN)  
- Decision Trees  
- Random Forest  
- Support Vector Machines (SVM)  
<br/>

## The Data
The original dataset contained:  
- 34 landmarks per face  
- 18 subjects  
- 4 expressions (Happy, Sad, Surprised, Disgusted)  
- 7 time frames per expression  

*Note: Due to privacy and size restrictions, the data set in this project is not included. The code is provided for reference, but can't run without the original data.
The provided code is for reference only* 

<br/>

## Requirements:
R version 4.5.1 (2025-06-13 ucrt)

Packages used:
- geomorph
- ggplot2
- randomForest
- e1071
- xgboost
- nnet
- rpart

Some of the results are displayed in the results folder.


MSc Dissertation Project by Fionn Johnson, University of Kent, 2025.
