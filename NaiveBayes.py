import numpy as np
import pandas as pd

class NaiveBayes():
    def __init__(self):
        pass
        
    def probability_tables(self):
        df = pd.concat([self.X, self.y], axis=1)
        
        probability_tables = []
        for col in self.attr_col:
            table = pd.DataFrame({'Count' : df[[col, self.target_col]].groupby([col, self.target_col]).size()}).reset_index()
            probability_tables.append(table)
    
        return dict(zip(self.attr_col, probability_tables))
    
    def fit(self, X, y):
        self.X = X
        self.y = y
        self.target_vals = np.sort(y.unique())        
        self.attr_col = X.columns
        self.target_col = y.name
        self.probability_tables = self.probability_tables()
        
    def predictor_prior_probability(self, key, val):
        temp_table = self.probability_tables[key]
        predictor_prior_probability = temp_table[temp_table[key]==val]['Count'].sum()/temp_table['Count'].sum()        
        return predictor_prior_probability
    
    def class_prior_probability(self, target_val):
        return self.y[self.y==target_val].count()/self.y.count()
    
    def likelihood(self, key, val, target_val):        
        temp_table = self.probability_tables[key]
        numerator = np.sum(temp_table[(temp_table[key]==val) & (temp_table[self.target_col]==target_val)]['Count'])
        denominator = np.sum(temp_table[temp_table[self.target_col]==target_val]['Count'])
        return numerator/denominator
        
    def predict(self, search):
        probabilities = []
        
        for t in self.target_vals:
            numerator = []
            denominator = []
            numerator.append(self.class_prior_probability(t))
            for key, val in search.items():
                numerator.append(self.likelihood(key, val, t))
                denominator.append(self.predictor_prior_probability(key, val))
            
            probabilities.append(np.prod(numerator)/np.prod(denominator))
        
        return dict(zip(self.target_vals, probabilities))


df = pd.read_csv('test_dataset.csv')

nb = NaiveBayes()
X = df[df.columns[:-1]]
y = df[df.columns[-1]]

question = {'Outlook': 'Sunny', 
            'Temperature': 'Cool', 
            'Humidity': 'High', 
            'Windy': True}
 
nb.fit(X,y)
nb.predict(question)