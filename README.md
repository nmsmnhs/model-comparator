# model.compare

A web app that automates ML model evaluation. Upload any CSV, pick a target column, 
and get instant comparisons across Logistic Regression, Decision Tree, and Random Forest.

## How to use it

- Upload a CSV dataset through the browser
- Select which column is your target 
- Automatically preprocesses the data (encoding, splitting, handling binary/multiclass targets)
- Trains 3 models and evaluates them on accuracy, precision, confusion matrix, and ROC-AUC
- Displays results as interactive charts

<img width="1919" height="956" alt="image" src="https://github.com/user-attachments/assets/767a4479-0435-491f-8f66-bd6a22cc9002" />
<img width="1897" height="1079" alt="image" src="https://github.com/user-attachments/assets/40eebfd7-6dbe-4599-924e-e4202dd148d8" />
<img width="1898" height="1079" alt="image" src="https://github.com/user-attachments/assets/204e8e05-1b0a-4ee9-afea-d50921ad25e2" />


## Tech Stack

- **Backend:** Python, Flask, scikit-learn, pandas
- **Frontend:** HTML, CSS, JavaScript, Chart.js

## Notes

- Uploaded datasets are deleted immediately after analysis
- Multiclass targets are automatically collapsed to binary
- ROC curves only appear for binary classification problems
