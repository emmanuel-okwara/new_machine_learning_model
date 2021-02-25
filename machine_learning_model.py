from numpy import inner
import pandas as pd
import os 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

files_location = None
for i,j,k in os.walk('C:\\Users\\okwar\\Pictures\\new_machine_learning_model'): #searching for the csv files that we are looking for 
    files_location = k[0:-1] # intializing the locations of the data 
#this is where the file structure is created 
data_file =['file{}'.format(x) for x in range(len(files_location))] 

# this is where the data is loaded onto the system.
datasets_in_folders = [pd.read_csv(x) for x in files_location] 

#assigning file structures to data in the list.
files = {x:y for x in data_file for y in datasets_in_folders} 

#joining the data 
data = pd.DataFrame(files[data_file[0]]).append(files[data_file[1]]).append(files[data_file[2]]).append(files[data_file[3]]).append(files[data_file[4]])
#print(data.columns)


#feature engineering
#renaming the columns 
renaming_columns = {'Overall rank':'OR', 'Country or region':'CNT/REG', 'Score':'SCR',
       'Social support':'SS', 'Healthy life expectancy' :'LFE_EXP',
       'Freedom to make life choices':'FOC', 'Generosity':"GEN",
       'Perceptions of corruption':"POC"}


#initilizing the data 
new_data = data.copy().rename(columns=renaming_columns)

#plotting to better understand the data
#plt.figure(figsize=(12,5,))
#plt.autoscale(axis=['both','x','y'])
#sns.lineplot(x=new_data['CNT/REG'],y=new_data['GEN'])
#plt.show()
missing_cols = [x for x in new_data.columns if new_data[x].isnull().any()]
#[] there is no columns with missing data

#spilting the data into our training and testing data
Y = new_data['GEN']
X = new_data.drop('GEN',axis=1)


x_train,x_test,y_train,y_test =train_test_split(X,Y,train_size=0.8,test_size=0.2)

#functions for working with the data 
def returns_the_training_data(x):
    fun_data = x.copy()
    num_cols = [col for col in fun_data.columns if fun_data[col].dtype in ['int64','float64']]
    obj_col = [col for col in fun_data.columns if fun_data[col].dtype == 'object']

    if 0 == int(False):
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import OneHotEncoder

    #imputing the numerical data 
    imputer = SimpleImputer(strategy = 'mean')
    encoder = OneHotEncoder(handle_unknown = 'ignore',sparse = False)

    x_imputed_data = pd.DataFrame(imputer.fit_transform(fun_data[num_cols]))
    x_imputed_data.columns = fun_data.columns 

    #encoding the categorical data 
    x_encoded_data = pd.DataFrame(encoder.fit_transform(fun_data[obj_col]))
    x_encoded_data.index = fun_data.index

    #real_train_data = x_imputed_data.merge(x_encoded_data)
    #return real_train_data
    return num_cols,obj_col

print(returns_the_training_data(x_train))






