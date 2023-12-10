# dpagoj-ML.3

Project name: Lenders Club Loan Grade and Subgrade Defining Application
Description: The goal of this project is to create a web service for predicting Lenders Club potential client risk Grade or Subgrade.
Project content:
1.	To analyze what were the potential criteria for rejecting the loan in the past.
2.	Practice identifying opportunities for data analysis, raising hypothesis, and formulating research tasks.
3.	Practice performing EDA, statistical inference, and prediction.
4.	Practice visualizing data.
5.	Practice machine learning modeling techniques.
6.	Practicing deploying the model for production in Google Cloud Service.
The project is twofold. The first part consists of 
1.	Jupiter notebook file ‘Lending.ipynb’ – where most of the analysis takes place. 
2.	In addition, there is a file ‘lender_functions.py’ where I place larger functions to safe the notebook space.
The second part consists of:
1.	Model deployment files – main.py, 
2.	Model file - model_steps.pkl, 
3.	app.yaml,
4.	requirements.txt files. 
5.	I also added a test.py file to test the service. 
Tu run the notebook, you need to store all the files in the same folder in your machine.
To test the service, you only need to run the test.py file.
The link of the service is: 
app_url = "https://gradepredict-400511.lm.r.appspot.com/predict" (I have disabled it for now so that GCS would not charge).
You are free to suggest any changed on how the project could be improved.
The original source of the file: link.
After downloading the datasets, name accept.csv and reject.csv. Please them in the same folder where the main Jupyter notebook file.
The project is not bind by any license agreement.
