# *********PLACEMENT PREDICTION MODEL ************


# IMPORTING REQUIRED LIBRARIES

import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt


# LOADING DATA FROM CSV FILE

data = pd.read_csv('placement.csv')

# print(data.head())


# sns.scatterplot(x=data["cgpa"],y=data["resume_score"],hue=data["placed"])     ***plotting loaded data using sns


 # SEPERATING FEATURES AND OUTPUT 
  
x = data.drop(["placed"],axis="columns")
y = data.placed
# print(x)



# CREATION OF PERCEPTRON

from sklearn.linear_model import Perceptron
p = Perceptron()

# TRAINING THE PERCEPTRON MODEL

p.fit(x,y)
# print(p.coef_)
# print(p.intercept_)



# PREDICTING RESULT FROM UNKNOWN DATA

result = p.predict([[5.62,6.52]])
if result==1:
    print("You will be placed.")
else:
    print("Better luck! Next time.") 



# VISUALIZING MACHINE'S INTELLIGENCE

from mlxtend.plotting import plot_decision_regions
plot_decision_regions(x.values,y.values,clf=p,legend=2)

plt.show()

