# Spam_Email_Detector

### Introduction:
This program is build with python, the related modules used in this project including CountVectorize, TfidfVectorize, SelectKBest, BernoulliNB, LogisticRegression, RandomForestClassifier and metrics in Sklearn.

Data:
33k emails

Data Preprocessing:
* Punctuation cleaning
* Replace url/phone number/number/attachment file name with corresponding replacer.

Model:
* 1, Processed raw data of email text into word tokens based on Count Vectorizer, TF-IDF in the ScikitLearn module.
* 2, Selected features using Chi-Square selection and max number limitation.
* 3, Built the models with Naive Bayes, Decision Tree, Logistic Reggression and Random Forest
* 4, Developed own evaluate metrics based on ROC and confusion matrix, by given penalty score to false positive.
* 5, Visualized the evaluation result with Matplotlib module
