import pandas as pd
import quandl
import math

import numpy as np
from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = quandl.get('WIKI/GOOGL')

df = df[['Adj. High','Adj. Low','Adj. Open','Adj. Close', 'Adj. Volume']]

df ['HL_PCT'] = ((df['Adj. High'] - df['Adj. Low']) / df['Adj. Low']) * 100
df ['CHANGE_PCT'] = ((df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open']) * 100

df = df[['Adj. Close', 'HL_PCT', 'CHANGE_PCT', 'Adj. Volume']]

forecast_column = 'Adj. Close'

df.fillna('-99999', inplace=True)

# forecast_out = int(math.ceil(0.1 * len(df)))
forecast_out = 10

# we create a label column that is actually the forecast_column but forecast_out days into the future.
df['label'] = df[forecast_column].shift(-forecast_out)

# now since we added labels as a shift, we will have rows that have no labels. These will be the last forecast_out rows.
# here we are dropping these rows. These rows ain't loyal.
df.dropna(inplace=True)

# print(forecast_out)

X = np.array(df.drop(['label'], 1))
y = np.array(df['label'])

X = preprocessing.scale(X)

# if we didn't drop the rows without labels earlier, this line could be used to do that.
# X = X[:-forecast_out + 1]

# print(len(X), len(y))

# now we will split out data into training and testing data.
# test_size defines what portion of the data we want to use for testing,
# the train-test-split function randomly will pick out test_size portion of entries.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9)

clf = LinearRegression()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test);

print(accuracy)