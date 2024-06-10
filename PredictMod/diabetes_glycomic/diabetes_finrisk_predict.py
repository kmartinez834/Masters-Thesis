"""
File:           diabetes_finrisk_predict.py
Author:         Karina Martinez
Version:        1.0
Description:    The script loads the pre-trained XGBoost Classifier from the diabetes_finrisk_classifier.pkl file, 
                reads an input file with plasma protein N-glycome levels, and outputs whether the patient is predicted 
                to develop type 2 diabetes or remain normoglycaemic within 10 years.

                This script accepts a filename argument:
                Usage: diabetes_finrisk_predict.py [options] 

                Options:
                    -h, --help                      show this help message and exit    
                    -f FILENAME, --file=FILENAME    patient input csv file [default: patient_input.csv]
"""

import joblib
import pandas as pd
from optparse import OptionParser

def preprocess_data(df):
    # Combine features with same structure
    glycan_dict = {
        "GP32_33":["GP32","GP33"],
        "GP38_39":["GP38","GP39"],
        "GP41_43":["GP41","GP42","GP43"]
               }
    
    # Create dataframe with combined features
    df_combined = pd.DataFrame()
    for key,value in glycan_dict.items():
        df_combined[key] = df[value].sum(axis=1)

    # Join combined features with original dataframe, drop individual peaks included in combined features
    df_out = df.join(df_combined)
    df_out = df_out.drop(columns=["GP32","GP33","GP38","GP39","GP41","GP42","GP43"])
    return df_out


###############################

def main():

    usage = "usage: %prog [options] "
    parser = OptionParser(usage=usage)
    parser.add_option("-f","--filename",action="store",dest="filename",default="patient_input.csv",help="patient input csv file [default: %default]")
    (options,args) = parser.parse_args()

    filename = options.filename

    # Open the pickle file
    model = joblib.load("diabetes_finrisk_classifier.pkl")

    # Get patient input data
    input_file = pd.read_csv(filename, sep=",")
    processed_df = preprocess_data(input_file)

    pred = model.predict(processed_df)

    outcome = 'Type 2 diabetes' if pred == 1 else 'Normoglycaemic'
        
    print(f"Prediction based on plasma protein N-glycome: {outcome}")


if __name__ == "__main__":
    main()