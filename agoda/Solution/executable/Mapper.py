#!/usr/bin/env python
# coding: utf-8

#import all required packages for this analysis
import pandas as pd
import os
import base64
import Levenshtein as lev
from fuzzywuzzy import process
from fuzzywuzzy import fuzz

# helper function to do preprocessing on string before fuzzy match 
def lowercase_removespace(df, col):
    '''
    Cleaning the dataset for matching algo. Following steps:
    make all lowercase -> remove spaces -> (remove comma, period, punctuation, etc)
    -> encode as utf-8
    '''
    df[col] = df.loc[:, col].map(lambda x: x.lower() if isinstance(x, str) else x).\
    map(lambda x: x.replace(' ','') if isinstance(x,str) else x).\
    map(lambda x: x.replace(',', '') if isinstance(x,str) else x).\
    map(lambda x: x.replace('.', '') if isinstance(x,str) else x)

    
def encode_to_utf8(df, col):
    '''encode string to utf-8 for consistency'''
    df[col] = df[col].map(lambda x: base64.b64encode(x.encode('utf-8')))
    
def decode_to_str(df, col):
    df[col] = df[col].map(lambda x: base64.b64decode(x))


def mapping(df_partner1, df_partner2, model = 'lev'):
    '''
    mapping function which calculate best match for hotels in Partners1 sheet with Partners2. 
    It takes Partners1 and Partners2 data as input and returns best match hotel from partner2 
    for each hotel in partner1 along with matching score

    Starts with checking for exact match first. model input takes two values ['lev', 'fuzzy']. 
    Default is Levenshtein distance model to map keys
    '''

    output = []
    country_list = [x for x in df_partner1['p1.country_code']]
    country_list = list(set(country_list))
    for country_code in country_list:
        partner1 = df_partner1[df_partner1['p1.country_code'] == country_code]
        partner2 = df_partner2[df_partner2['p2.country_code'] == country_code]
        for str1 in partner1['p1.mapping_key']:
            matching_score = []
            best_match = []
            for str2 in partner2['p2.mapping_key']:
                if str1 == str2:
                    ratio = 1.0
                    matching_score.append(ratio)
                    best_match.append(str2)
                else:
                    if model == 'lev':
                        ratio = lev.ratio(str1, str2)
                        matching_score.append(ratio)
                        best_match.append(str2)
                    elif model == 'fuzzy':
                        ratio = fuzz.token_set_ratio(str1, str2)/100.0
                        matching_score.append(ratio)
                        best_match.append(str2)                    
                        
            max_index = matching_score.index(max(matching_score))
            max_matching_score = matching_score[max_index]
            best_match_hotel = best_match[max_index]
            p1_key = partner1.loc[partner1['p1.mapping_key'] == str1,['p1.key']].iloc[0,0]
            p2_key = partner2.loc[partner2['p2.mapping_key'] == best_match_hotel, ['p2.key']].iloc[0,0]
            output.append([str1, p1_key, best_match_hotel, p2_key, 
                           country_code, max_matching_score]) 
    return output

def main():
    #read raw input data 
    file = input('Please enter the input file name: \n')
    #file = '/Users/santoshkumar/Downloads/agoda/mappinghotelsdataset.xlsx'
    #file = os.path.join(input_file_path, 'mappinghotelsdataset.xlsx')

    raw_df_partner1 = pd.read_excel(file, sheet_name='Partner1', keep_default_na=False)
    df_partner1 = raw_df_partner1
    raw_df_partner2 = pd.read_excel(file, sheet_name = 'Partner2', keep_default_na=False)
    df_partner2 = raw_df_partner2

    
    #Applying our winning model with optimal threshold on Partner1 sheet
    #Applying helper functions to clean data for string matching 

    lowercase_removespace(df_partner1, 'p1.hotel_name')
    lowercase_removespace(df_partner1, 'p1.city_name')
    lowercase_removespace(df_partner1, 'p1.hotel_address')
    lowercase_removespace(df_partner1, 'p1.country_code')

    lowercase_removespace(df_partner2, 'p2.hotel_name')
    lowercase_removespace(df_partner2, 'p2.city_name')
    lowercase_removespace(df_partner2, 'p2.hotel_address')
    lowercase_removespace(df_partner2, 'p2.country_code')

     
    #convert hotel name and address to avoid any funny behaviour
  
    df_partner1 = df_partner1.astype({'p1.hotel_name': str, 'p1.city_name': str, 'p1.hotel_address': str})
    df_partner2 = df_partner2.astype({'p2.hotel_name': str, 'p2.city_name': str, 'p2.hotel_address': str})

    
    #create a mapping key which is concatenation of hotel name and hotel address. 
    #This key will be used to calculate matching score 
    
    df_partner1['p1.mapping_key'] = df_partner1['p1.hotel_name'] + df_partner1['p1.city_name']
    df_partner2['p2.mapping_key'] = df_partner2['p2.hotel_name'] + df_partner2['p2.city_name']

    encode_to_utf8(df_partner1, 'p1.mapping_key')
    encode_to_utf8(df_partner2, 'p2.mapping_key')

    best_mapping_file = mapping(df_partner1, df_partner2)
    df_mapping = pd.DataFrame(best_mapping_file)
    df_mapping.columns = ['p1.mapping_key','p1.key', 'p2.mapping_key','p2.key','country_code','score']
    

    submission_df = df_mapping[df_mapping['score'] >= .61]
    submission_df = submission_df[['p1.key', 'p2.key']]
    submission_df.to_csv('mappings.csv', index = False)
    print('Successfully created mapping file!\nYour output file named "mapping.csv" is saved at {}'.format(os.getcwd()))


if __name__=='__main__':
    main()
