import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline
base = 'C:/Users/eaa-h/ML/FirstHW/Lab_EnteringML/'
data = pd.read_csv(base + 'organisations.csv')
features = pd.read_csv(base + 'features.csv')
rubrics = pd.read_csv(base + 'rubrics.csv')
fig, ax = plt.subplots(figsize=(5, 2.7), layout='constrained')
ax.set_xlabel('Average Bill')  # Add an x-label to the Axes.
ax.set_ylabel('Number of rest')
plt.xlim(0, 2500)
valuesInMSKDELTA = data[data['city'] == 'msk']
valuesInMSK = valuesInMSKDELTA[data['average_bill'] <= 2500]
valuesInSPBDelta = data[data['city'] == 'spb']
valuesInSPB = valuesInSPBDelta[data['average_bill'] <= 2500]
print(len(valuesInMSK['average_bill'].dropna())+len(valuesInSPB['average_bill'].dropna()))
n, bins, patches = ax.hist(valuesInMSK['average_bill'].dropna(), 5, facecolor='C0', alpha=0.75)
n2, bins2, patches2 = ax.hist(valuesInSPB['average_bill'].dropna(), 5, facecolor='C1', alpha=0.75)
# Difference between average bill in SPB and MSK
#average_bills = data.groupby('city')['average_bill'].mean()
#average_bill_msk = average_bills['msk']
#average_bill_spb = average_bills['spb']

#print("Difference:",average_bill_msk-average_bill_spb)
average_bill_msk = data[data['city'] == 'msk']['average_bill'].mean()
average_bill_spb = data[data['city'] == 'spb']['average_bill'].mean()

print("Difference:", average_bill_msk - average_bill_spb)
# -----ML------
#clean_data_train, clean_data_test = train_test_split(
 #   clean_data, stratify=clean_data['average_bill'], test_size=0.33, random_state=42)

plt.show()