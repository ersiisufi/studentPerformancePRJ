import pandas as pd
from sklearn.model_selection import train_test_split

def load_data():
    df = pd.read_csv('/Users/admin/Documents/ML projects/student performance/data/Students_perf')
    return df

def preprocess(df):
    X = df.drop('math score', axis=1)
    y = df['math score']
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

