import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

# Load two datasets
Observations = pd.read_csv('/content/observations.csv')
Species = pd.read_csv('/content/species_info.csv')

print(Observations.head())
Species.head()

Observations.info()

Species.info()

print(Observations.describe(include='all'), '\n')
print(Species.describe(include='all'))

sns.histplot(Observations.observations)

sns.countplot(x = 'conservation_status', data=Species, hue='category')
plt.legend(loc='upper right')

plt.bar('park_name', 'observations', data=Observations)
ax = plt.subplot()
ax.tick_params(labelrotation=90)

Combined = pd.merge(Observations, Species, how='inner')
Combined.head()

Combined.info()

sns.countplot(x='conservation_status', data=Combined, hue='park_name')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics

##1.data slection---------------------------------------------------
#def main():
dataframe=Combined

print("---------------------------------------------")
print()
print("Data Selection")
print("Samples of our input data")
print(dataframe.head(10))
print("----------------------------------------------")
print()

#2.pre processing--------------------------------------------------
#checking  missing values
print("---------------------------------------------")
print()
print("Before Handling Missing Values")
print()
print(dataframe.isnull().sum())
print("----------------------------------------------")
print()

print("-----------------------------------------------")
print("After handling missing values")
print()
dataframe_2=dataframe.fillna(0)
print(dataframe_2.isnull().sum())
print()
print("-----------------------------------------------")

from sklearn.preprocessing import LabelEncoder

number = LabelEncoder()

dataframe['scientific_name'] = number.fit_transform(dataframe['scientific_name'].astype(str))
dataframe['park_name'] = number.fit_transform(dataframe['park_name'].astype(str))
dataframe['category'] = number.fit_transform(dataframe['category'].astype(str))
dataframe['common_names'] = number.fit_transform(dataframe['common_names'].astype(str))
dataframe['conservation_status'] = number.fit_transform(dataframe['conservation_status'].astype(str))

dataframe

X=pd.DataFrame(dataframe, columns =['scientific_name','park_name','observations','common_names','conservation_status'])
y=pd.DataFrame(dataframe, columns =['category'])

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.20,random_state = 42)

from sklearn.ensemble import RandomForestClassifier

rf= RandomForestClassifier(n_estimators = 100)
rf.fit(x_train, y_train)
rf_prediction = rf.predict(x_test)
Result_3=accuracy_score(y_test, rf_prediction)*100
from sklearn.metrics import confusion_matrix

print()
print("---------------------------------------------------------------------")
print("Random Forest")
print()
print(metrics.classification_report(y_test,rf_prediction))
print()
print("Random Forest Accuracy is:",Result_3,'%')
print()
print("Confusion Matrix:")
cm2=confusion_matrix(y_test, rf_prediction)
print(cm2)
print("-------------------------------------------------------")
print()
import matplotlib.pyplot as plt
import seaborn as sns

sns.heatmap(cm2, annot = True, cmap ='plasma',
        linecolor ='black', linewidths = 1)
plt.show()

import keras.backend as K
def get_f1(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

import keras
from keras.models import Sequential
from keras.layers import Dense

import matplotlib.pyplot as plt

scaler = MinMaxScaler()
X_train = scaler.fit_transform(x_train)
X_test = scaler.transform(x_test)

input_shape = X_train.shape[1:]
num_classes = y_train.shape[1]

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D, MaxPool1D, Flatten, Input
inp =  Input(shape=(5,1))
conv = Conv1D(filters=2, kernel_size=2)(inp)
pool = MaxPool1D(pool_size=2)(conv)
flat = Flatten()(pool)
dense = Dense(1)(flat)
model = Model(inp, dense)
model.compile(loss='mse', optimizer='adam',metrics=['acc'])

print(model.summary())
model.fit(X_train,y_train,epochs = 10,batch_size = 20)

model.compile(loss='mae', optimizer='adam', metrics=['acc'])

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=64)

loss, accuracy = model.evaluate(X_test, y_test, verbose=2)

print("Accuracy:")
print(accuracy)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['Training', 'Testing'], loc='upper left')
plt.savefig("Accuracy_img.png")

rf_prediction=np.array(rf_prediction)
rf_prediction.dtype
rf_prediction=np.uint8(rf_prediction)

import numpy as np
y = pd.Series(rf_prediction)
print(y)
y.to_csv('/content/out.csv')

inp=int(input('Enter the Species Id '))
if (rf_prediction[inp]==6):
    print("Species Detected ")
else:
    print("Species Not Detected")