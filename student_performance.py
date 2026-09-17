import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

def build_dataset():
    return pd.DataFrame({'study_hours':[1,2,3,4,5,6,7,8,2,5,3,7,4,6,1], 'attendance':[45,50,58,65,72,78,85,92,48,80,62,88,68,76,40], 'result':[0,0,0,1,1,1,1,1,0,1,0,1,1,1,0]})

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--study-hours',type=float,required=True)
    parser.add_argument('--attendance',type=float,required=True)
    a=parser.parse_args()
    if a.study_hours<0 or not 0<=a.attendance<=100: raise SystemExit('Invalid input.')
    df=build_dataset(); X=df[['study_hours','attendance']]; y=df['result']
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.3,random_state=42,stratify=y)
    model=DecisionTreeClassifier(max_depth=3,random_state=42); model.fit(Xtr,ytr)
    pred=model.predict(Xte); print(f'Test accuracy: {accuracy_score(yte,pred)*100:.2f}%')
    print(classification_report(yte,pred,zero_division=0))
    sample=pd.DataFrame({'study_hours':[a.study_hours],'attendance':[a.attendance]})
    print('Prediction:', 'PASS' if model.predict(sample)[0] else 'FAIL')

if __name__=='__main__': main()
