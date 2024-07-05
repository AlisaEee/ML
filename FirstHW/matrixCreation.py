import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
#%matplotlib inline

#from sklearn.model_selection import train_test_split
def construct_matrix(first_array, second_array):
    result = np.dstack((first_array, second_array))
    # так тоже можно : return np.vstack([first_array, second_array]).T # <- your first right code here
    return result
print(construct_matrix(np.array([1,2,5,6]),np.array([3,4,7,8])))