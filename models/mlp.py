import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from keras.models import Sequential 
from keras.layers import Dense 

MAX_FEATURES = 93
preprocessed_data = pd.read_csv('dataset/preprocessed_binary_dataset.csv')

features = preprocessed_data.iloc[:,0:MAX_FEATURES].values 
target = preprocessed_data[['intrusion']].values 

X_train, X_test, y_train, y_test = train_test_split(features,target, test_size=0.25, random_state=42)
X_train=np.asarray(X_train).astype(np.float32)
X_test=np.asarray(X_test).astype(np.float32)

mlp = Sequential() # creating model

# adding input layer and first layer with 50 neurons
mlp.add(Dense(units=50, input_dim=X_train.shape[1], activation='relu'))
# output layer with sigmoid activation
mlp.add(Dense(units=1,activation='sigmoid'))

mlp.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy","f1_score","precision", "recall"])

mlp.summary()

history = mlp.fit(X_train, y_train, epochs=100, batch_size=5000,validation_split=0.2)

test_results = mlp.evaluate(X_test, y_test, verbose=1, return_dict=True)
###print(f'Test results - Loss: {test_results[0]} - Accuracy: {test_results[1]*100}')
print(test_results)
