import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import os
import codecs
import re
from sklearn.metrics import confusion_matrix


score_dict = {'fp' : -1000, 'tp' : 100, 'fn' : -100, 'tn' : 0}



def plotAUC(truth, pred, lab):
    fpr, tpr, thresholds = roc_curve(truth, pred)
    roc_auc = auc(fpr, tpr)
    c = (np.random.rand(), np.random.rand(), np.random.rand())
    fig = plt.figure(dpi=200)
    plt.plot(fpr, tpr, color=c, label= lab+' (AUC = %0.4f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC of %s'%lab)
    plt.legend(loc="lower right")

def getScoreForGridSearch(estimator, X, y):
    score_dict = {'fp' : -1000, 'tp' : 100, 'fn' : -100, 'tn' : 0}
    fpr, tpr, thresholds = roc_curve(y, estimator.predict_proba(X)[:,1])
    scores = fpr*score_dict['fp'] + \
            (1-fpr)*score_dict['tn'] + \
            tpr*score_dict['tp'] + \
            (1-tpr)*score_dict['fn']
    return max(scores)

def getScore(truth, pred, score_dict = {'fp' : -1000, 'tp' : 100, 'fn' : -100, 'tn' : 0}):
    fpr, tpr, thresholds = roc_curve(truth, pred)
    scores = fpr*score_dict['fp'] + \
            (1-fpr)*score_dict['tn'] + \
            tpr*score_dict['tp'] + \
            (1-tpr)*score_dict['fn']
    return thresholds, scores

def plotScore(truth, pred, lab):
    """
    this function will plot score function value
    according to customized score_diction
    input: truth, prediction, label for legend
    return: max score, best threshold according to score function
    """
    fpr, tpr, thresholds = roc_curve(truth, pred)
    scores = fpr*score_dict['fp'] + \
            (1-fpr)*score_dict['tn'] + \
            tpr*score_dict['tp'] + \
            (1-tpr)*score_dict['fn']
    best_thresholds = thresholds[np.argmax(scores)]
    fig = plt.figure(dpi=200)
    plt.plot(thresholds, scores, label= lab+' Best score = \
             %0.4f\nthreshold = %0.4f' % (max(scores),best_thresholds))
    plt.xlabel('threshold')
    plt.ylabel('score')
    plt.title('Score')
    plt.legend(loc="lower right")
    return max(scores),best_thresholds


def confusion_mtx(truth, pred, best_threshold):
    predicted = (pred>= best_threshold).astype('int')
    cm = confusion_matrix(truth,predicted)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm = pd.DataFrame(cm)
    cm.columns = ['POS(Truth)','Neg(Truth)']
    cm.index = ['POS(Pred)','Neg(Pred)']
    return cm


def gather_auc(X_test, Y_test, logreg, nb, dt, rf):
    from sklearn import metrics
    tprs=[]
    fprs=[]
    roc_labels=[]

    ## get roc curve with predict_proba
    fpr, tpr, thresholds = metrics.roc_curve(Y_test, logreg.predict_proba(X_test)[:,1])
    tprs.append(tpr)
    fprs.append(fpr)
    label_b_l=auc(fpr,tpr)
    roc_labels.append(" Binary with Logistic. AUC--"+str(label_b_l))


    fpr, tpr, thresholds = metrics.roc_curve(Y_test, nb.predict_proba(X_test)[:,1])
    tprs.append(tpr)
    fprs.append(fpr)
    label_b_l=auc(fpr,tpr)
    #label_b_l=metrics.roc_auc_score(model.predict(X_test_features_b), Y_test)
    roc_labels.append(" Binary with Naive Bayes. AUC--"+str(label_b_l))


    fpr, tpr, thresholds = metrics.roc_curve(Y_test, dt.predict_proba(X_test)[:,1])
    tprs.append(tpr)
    fprs.append(fpr)
    label_b_l=auc(fpr,tpr)
    #label_b_l=metrics.roc_auc_score(model.predict(X_test_features_b), Y_test)
    roc_labels.append(" Binary with decision tree. AUC--"+str(label_b_l))


    fpr, tpr, thresholds = metrics.roc_curve(Y_test, rf.predict_proba(X_test)[:,1])
    tprs.append(tpr)
    fprs.append(fpr)
    label_b_l=auc(fpr,tpr)
    #label_b_l=metrics.roc_auc_score(model.predict(X_test_features_b), Y_test)
    roc_labels.append(" Binary with Randomforest. AUC--"+str(label_b_l))
    return tprs, fprs, roc_labels

def plot_auc_together(tprs, fprs, roc_labels):
    #plt.rcParams['figure.figsize'] = 12, 12
    fig = plt.figure(dpi=200)
    for fpr, tpr, roc_label in zip(fprs, tprs, roc_labels):
        plt.plot(fpr, tpr, label=roc_label)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend(loc=4)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.show()
              
              
def cleanText(text):
    # This function takes in a text string and cleans it 
    # by keeping only alphanumeric and common punctuations
    # Returns the cleaned string
    clean_text = text.replace('\n',' ').replace('\r',' ')
    clean_text = re.sub(r'[^a-zA-Z0-9.:!? ]',' ',clean_text)
    return clean_text



def readEmailAsDataFrame():

    def cleanText(text):
        # This function takes in a text string and cleans it 
        # by keeping only alphanumeric and common punctuations
        # Returns the cleaned string
        clean_text = text.replace('\n',' ').replace('\r',' ')
        clean_text = re.sub(r'[^a-zA-Z0-9.:!? ]',' ',clean_text)
        return clean_text

    def readEmail(path):
        # This function takes a path to an email text file
        # Returns a tuple of the subject line and the body text
        with codecs.open(path, "r",encoding='utf-8', errors='ignore') as f:
            subject = cleanText(f.readline()[9:])
            body = cleanText(f.read())
            return [subject, body]

    subjects = []
    bodys = []
    spam = []
    # read hams
    for i in range(1, 6+1):
        folder_name = "enron"+str(i)
        for filename in os.listdir(folder_name+"/ham"):
            if filename.endswith(".txt"):
                subject, body = readEmail(folder_name+"/ham/"+filename)
                subjects.append(subject)
                bodys.append(body)
                spam.append(0)
        # read spams
        for filename in os.listdir(folder_name+"/spam"):
            if filename.endswith(".txt"):
                subject, body = readEmail(folder_name+"/spam/"+filename)
                subjects.append(subject)
                bodys.append(body)
                spam.append(1)
    data = pd.DataFrame()
    data['subject'] = subjects
    data['body'] = bodys
    data['spam'] = spam

    # store using pickle
    data.to_pickle("enron_email.df")

    return data

