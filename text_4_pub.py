# -*- coding: utf-8 -*-
"""
@author: Art
The class creates and fit a prediction model (based on linear SVM) for text publication.
The model makes a recommendation of site for text posting. Please, be careful with encoding.
Usage:
>> hubr0geek = Site_4_publication()
>> hubr0geek.predict('my_input.txt') 

"""

import numpy as np
import pandas as pn
import sklearn.cross_validation
import matplotlib.pyplot as plt
import sklearn.preprocessing 
import sklearn.linear_model
import sklearn.metrics
import sklearn.feature_extraction
import sklearn.svm
from sklearn.externals import joblib
import codecs

class Site_4_publication:
    def __init__(self, clf_file=None):
        self.y = []
        self.text = []
        self.X_TV = sklearn.feature_extraction.text.TfidfVectorizer()            
        self.text_load()
        if not clf_file:
            self.classification_train()
        else:
            self.load(clf_file)
            
    def text_load(self):
        self.text_open('habrahabr.json', 0)
        self.text_open('geektimes.json', 1)
        self.text_explore()
    
    def text_open(self, text_file, Geek_flag):
        text = pn.read_json(text_file)
        text['text'] = text['text'].replace('[^a-zA-Z0-9]', ' ', regex = True)
        for i in text['text']:
            #self.text = map(lambda x: x.lower(), self.text)
            self.text.append(i.lower())
        if Geek_flag == 1:
            self.y = self.y + [0]*len(text)
        else:
            self.y = self.y + [1]*len(text)
        
    def text_explore(self):
        
            
        
        
        self.X_dat = self.X_TV.fit_transform(self.text)
        
        
    def classification_train(self):
        X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(self.X_dat,\
                                      self.y, test_size=0.2, random_state=142)    
        self.clf = sklearn.svm.LinearSVC(C = 1.0, penalty='l2')
        self.my_svm = self.clf.fit(X_train, y_train)
        self.plot_rocauc(y_test, X_test)
        
        
        
    def classification_test(self, y_test= [], X_test= []):
        X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(self.X_dat, \
                                      self.y, test_size=0.1, random_state=684)
        return sklearn.metrics.roc_auc_score(y_test, self.my_svm.predict(X_test))      
        
    def save(self, file_name):
        joblib.dump(self.my_svm, file_name)
        print 'Save complete!'
   
    def load(self, file_name):
        self.my_svm = joblib.load(file_name)
        print 'Load complete!'
        
    def plot_rocauc(self, y_test, X_test):
        false_positive_rate, true_positive_rate, thresholds = sklearn.metrics.roc_curve(y_test, self.my_svm.decision_function(X_test))
        roc_auc = sklearn.metrics.auc(false_positive_rate, true_positive_rate)       
        plt.title('Receiver Operating Characteristic')
        plt.plot(false_positive_rate, true_positive_rate, 'b',
                 label='AUC = %0.2f'% roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0,1],[0,1],'r--')
        plt.xlim([-0.1,1.2])
        plt.ylim([-0.1,1.2])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()

        
        
        print 'Accuracy:', sklearn.metrics.accuracy_score(y_test, self.my_svm.predict(X_test))
        print 'Prescision:', sklearn.metrics.precision_score(y_test, self.my_svm.predict(X_test))
        print 'Recall:', sklearn.metrics.recall_score(y_test, self.my_svm.predict(X_test))
        print 'Roc_auc:', self.classification_test(y_test, self.my_svm.predict(X_test))

        
    def predict(self, new_text_file='input.txt'):
        infile = codecs.open(new_text_file, 'r', encoding='utf-8')
        new_text = infile.read()
        prediction = self.my_svm.predict(self.X_TV.transform([new_text]))
        if prediction[0] == 0:
            print 'Recomendation: Habrahabr'
                  
        else: 
            print 'Recomendation: Geektimes'
        infile.close()

if __name__ == '__main__':
    hubr0geek = Site_4_publication()
    #hubr0geek.predict('input.txt') 