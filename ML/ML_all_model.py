import os,random,sys
from sklearn.model_selection import train_test_split
# classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.inspection import permutation_importance
from sklearn.metrics import *
import seaborn as sns
import matplotlib.font_manager as font_manager
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler

def main(argv):
    random.seed(0)

    global folder 
    folder = 'all_model'
      
    if os.path.isdir(folder) is False:
        os.makedirs(folder)
        
    # data pre-processing
    # property.txt is a spaced-delimited file with the first column as the name(not used), and last column as stability label
    # all the other columns are features
    data,label,features = data_process('property.txt')

    # custom testing data settings
    # when this is on will use the trained model on the provide custom datset
    custom_test = False 
    if custom_test:
      data_custom,label_custtom,_ = data_process('property_custom.txt',desire_feat_ind=desire_feat_ind)



    f = open('{}/result.txt'.format(folder),'w')
    for model in ['dt','KNN','LR_l1','LR_l2','SVC','GNB','rf']:
    
       print(model)

       reports,clf = train_tune(data,label,model=model)

       f.write('\n{}\n'.format(model))
       for i in reports:
         f.write(str(i))


       if custom_test:
          predict_label = clf.predict(data_custom)
          correct = 0
          total = 0
          for count_i,i in enumerate(label_custom):
            if i == predict_label[count_i]:
               correct += 1
            total += 1
          print(total-correct)
          print('Testing accuracy for custom dataset: {:<4.2f}'.format(correct/total))
          f.write('Testing accuracy for custom dataset: {:<4.2f}, {:} ligands wrong'.format(correct/total,total-correct))
     

    f.close()
    
    

def train_tune(data,label,epoch=1,model='dt'):
    #train_score = []
    #test_score = []

    #X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2,random_state=0)
    #### Use oversampling
    # define oversampling strategy
    oversample = RandomOverSampler(sampling_strategy='minority')
    # fit and apply the transform
    X_over, y_over = oversample.fit_resample(data, label)
    X_train, X_test, y_train, y_test = train_test_split(X_over, y_over, test_size=0.2,random_state=0)

    if model == 'dt':
       tuned_parameters = [{'max_depth':list(range(1,20))}]
       clf = GridSearchCV(DecisionTreeClassifier(),tuned_parameters,scoring='accuracy',return_train_score=True)
    elif model == 'KNN':
       tuned_parameters = [{'n_neighbors':list(range(3,10))}]
       clf = GridSearchCV(KNeighborsClassifier(),tuned_parameters,scoring='accuracy',return_train_score=True)
    elif model == 'LR_l1':
       tuned_parameters = [{'C':[1e-3,1e-2,0.1,1,10,1e2,1e3],'penalty':['l1'],'solver':['saga'],'max_iter':[200],'multi_class':['ovr']}]
       clf = GridSearchCV(LogisticRegression(),tuned_parameters,scoring='accuracy',return_train_score=True)
    elif model == 'LR_l2':
       tuned_parameters = [{'C':[1e-3,1e-2,0.1,1,10,1e2,1e3],'penalty':['l2']}]
       clf = GridSearchCV(LogisticRegression(),tuned_parameters,scoring='accuracy',return_train_score=True)
    elif model == 'SVC':
       tuned_parameters = [{'C':[1e-3,1e-2,0.1,1,10,1e2,1e3],'gamma':['auto']}]
       clf = GridSearchCV(SVC(),tuned_parameters,scoring='accuracy',return_train_score=True)
    elif model == 'GNB':
       tuned_parameters = [{'var_smoothing':[1e-9]}]
       clf = GridSearchCV(GaussianNB(),tuned_parameters,scoring='accuracy',return_train_score=True)
    elif model == 'rf':
       tuned_parameters = [{'max_depth':list(range(1,20)),'bootstrap':[True],'oob_score':[True,False]}]
       clf = GridSearchCV(RandomForestClassifier(),tuned_parameters,scoring='accuracy',return_train_score=True)
    clf.fit(X_train,y_train)
    reports = []
    reports.append("Best parameters set found on development set:\n\n")
    reports.append(clf.best_params_)
    reports.append("\nGrid scores on development set:\n\n")

    means = clf.cv_results_["mean_test_score"]
    stds = clf.cv_results_["std_test_score"]
    means_train = clf.cv_results_["mean_train_score"]
    stds_train = clf.cv_results_["std_train_score"]
    for mean_train, std_train, mean, std, params in zip(means_train, stds_train, means, stds, clf.cv_results_["params"]):
        reports.append("%0.3f (+/-%0.03f) %0.3f (+/-%0.03f) for %r\n" % (mean_train, std_train * 2, mean, std * 2, params))

    reports.append("\nDetailed classification report:\n\n")
    reports.append("The model is trained on the full development set.\n")
    reports.append("The scores are computed on the full evaluation set.\n\n")

    y_true, y_pred = y_test, clf.predict(X_test)
    reports.append(classification_report(y_true, y_pred))
    reports.append('\n')
    clf = clf.best_estimator_
    
    return reports,clf



# can specified desire_feat_ind to get specific features only
def data_process(filename,desire_feat_ind=[]):
    data = []
    label = []
    with open(filename,'r') as f:
        for lc,lines in enumerate(f):
            fields = lines.split()
            if lc == 0: 
                if desire_feat_ind == []:
                    features = fields[2:-1] # including all feat, 0: name, 1: linker, -1: label
                else:
                    features = [ fields[_] for _ in desire_feat_ind]
                continue
            if desire_feat_ind == []:
                data.append([float(i) for i in fields[2:-1]])
            else:
                tmp_feat = [ float(fields[_]) for _ in desire_feat_ind]
                data.append(tmp_feat)
            if int(fields[-1]) == 0: 
               label.append(0)
            else:
               label.append(1)
            #label.append(int(fields[-1]))


    data = np.array(data)
    label = np.array(label)
    
    return data,label,features



if __name__ == "__main__":
    main(sys.argv[1:])
