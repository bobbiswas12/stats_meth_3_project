import csv
import numpy as np

def load_data(file_path):

    with open(file_path, "r", newline="") as file:

        #so I am reading the file as though each of its row is a dictionary with the columns names being the first row of the csv file (i.e. our variable names - id, gender, age, etc..
        reader = csv.DictReader(file)

        #enter the columns that we do not want to work with, here - 
        explanatory_var_excluded = ["id","gender","ever_married","work_type","Residence_type","smoking_status","stroke"]

        #now, how do we handle the 'N/A' values? the best I could think of is to replace them with either the mean of all the values or the median, for now, I am just going to go with the mean and for that, I will just maintain the mean of all the respective explanatory Observations in the form of a dictionary 'mean_X'

        #initializing the Mean_value dictionary mean_X ->
        mean_X = {};
        for col in reader.fieldnames:
            if col not in explanatory_var_excluded:
                mean_X[col] = 0

        #calculating the mean-values ->
        no_of_entries = 0
        for row in reader:
            for col in reader.fieldnames:
                if col not in explanatory_var_excluded and row[col] != "N/A":
                    value = row[col]
                    mean_X[col] += float(value)
            no_of_entries+=1

        #print(mean_X)
        for col in reader.fieldnames:
            if col not in explanatory_var_excluded:
                mean_X[col] /= no_of_entries 

        #print(mean_X)

        #now just forming the nxp matrix 'X' (we shall add the first column of '1' values later (its fine, i haven't forgotten about it))

    with open(file_path, "r", newline="") as file: #so apparently, the reader in which the dictionary is stored in gets exhausted after you are done looping through all of its key-values once so you must reopen it for another use

        reader = csv.DictReader(file)
            
        X = []
        Y = []
        for row in reader:
            features = []
            for col in reader.fieldnames:
                if col not in explanatory_var_excluded and row[col] not in ("N/A" , ""):
                    features.append(float(row[col]))
                elif col not in explanatory_var_excluded and row[col] in ("N/A",""):
                    features.append(mean_X[col])
            X.append(features)
            Y.append(row['stroke'])
        X = np.array(X)
        X_1 = np.ones((no_of_entries,1))
        X = np.concatenate((X_1,X), axis = 1)
        Y = np.array(Y,dtype=float)

    return X,Y
