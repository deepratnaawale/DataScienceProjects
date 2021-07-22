# Author: Deepratna Awale
# Date: 06/06/2021
# Note: To execute this file just run it using python 3 interpreter, please do read the README before executing. 

# imports
import pandas as pd
import requests
import time
import os
import io
import re
from datetime import datetime
import nltk
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer

nltk_stopwords = stopwords.words('english')

positives_file = 'Dictionaries/positive_words.csv'
negatives_file = 'Dictionaries/negative_words.csv'
stopword_file = 'Dictionaries/stopwords.csv'
complex_words_file = 'Dictionaries/complex_words.csv'
uncertainty_words_file = 'Dictionaries/uncertainty_words.csv'
constraining_words_file = 'Dictionaries/constraining_words.csv'


"""
EDGAR Files Downloader
"""

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    
    # Print New Line on Complete
    if iteration == total: 
        print()


# log events in file everytime files are downloaded
def log_event(file, string):
    with open(file, 'a+') as f:
        f.write(string)

# function to setup Directory structure if not available already
def make_directories(directories):
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print('These Directories were created: ', directories)
            return
    print('Required Directories already exist.')
    return

# downloads text file from EDGAR, cleans it using BS and returns a string object
def get_file(url):
    cleantext = BeautifulSoup(requests.get(url, allow_redirects=True).content, "lxml").text
    
    if (cleantext.find("Your Request Originates from an Undeclared Automated Tool") == -1):
        time.sleep(0.1)
        return cleantext
    elif cleantext == None:
        time.sleep(1)
        return get_file(url)
    else:
        time.sleep(1)
        return get_file(url)

# intakes urls from provided excel file whose path is in sources, saves the downloaded files to 'Extracted Data'
def get_data_from_sec_gov_archives(sources, destination):
    
    base_url = 'https://www.sec.gov/Archives/'
    log_file = 'Logs/' + datetime.now().strftime("%d-%m-%Y-%H-%M-%S") + '.txt'
    total = len(sources)
    count = 0
    
    print('Acquiring {} files.'.format(total))
    printProgressBar(0, total, prefix = 'Progress:', suffix = 'Complete', length = 50)
    
    for source in sources:
        content = get_file(base_url + source)
        
        with io.open(destination + source[-24:], 'w+', encoding="utf-8") as file:
            file.write(content)
        
        log_event(log_file, 'Wrote to File {} waiting 1 second(s).\n'.format(source[-24:]))
        log_event(log_file, str(total - count) + ' files remaining.\n')
        count += 1
        
        printProgressBar(count, total, prefix = 'Progress:', suffix = 'Complete', length = 50)
    
    log_event(log_file, 'Complete')   
    print('Data Acquired.\n Logs are now stored in {}'.format(log_file))
    
    


"""
Analyze Files
"""
# makes a stemmed word dictionary of positive words, negative words, complex words, stop words, uncertainty words and constraining words
# input should be path of positive word file, negative word file, stop word file, complex word file, uncertainty word file and constraining word file
# note that this function requires stemmed word list (I've done this to optimize the time required), instead of stemming words everytime, we store and load them
# output is a dictionary object
def get_word_dict(positives_file, negatives_file, stop_words_file, complex_words_file, uncertainty_words_file, constraining_words_file):
    
    
    print('Constructing stemmed word list dictionary.')
    positive_words_df = pd.read_csv(positives_file)
    negative_words_df = pd.read_csv(negatives_file)
    complex_words_df = pd.read_csv(complex_words_file)
    lm_stopwords_df = pd.read_csv(stop_words_file)
    uncertainty_words_df = pd.read_csv(uncertainty_words_file)
    constraining_words_df = pd.read_csv(constraining_words_file)


    word_dict = {
        'positive_words' : set(positive_words_df['Word']),
        'negative_words' : set(negative_words_df['Word']),
        'complex_words' : set(complex_words_df['Word']),
        'lm_stopwords' : set(lm_stopwords_df['Word']),
        'uncertainty_words' : set(uncertainty_words_df['Word']),
        'constraining_words' : set(constraining_words_df['Word'])
    }
    
    return word_dict


# removes anything that is not an alphabet using regex and removes nltk stopwords and LM stopwords
# tokenizes input in text into words
# returns a list of words in the sentence
def get_filtered_text(text, stopwords):
    clean_text = []
    text = re.sub('[^a-zA-Z]+', ' ', text) # keep only alphabets
    for word in nltk.tokenize.word_tokenize(text):
        if word not in stopwords:
            clean_text.append(word)
    return clean_text


# intakes polarity score and assigns a categorical lable based on inputs value
# returns a string in {'Mostly Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive'}
def categorize_sentiment(polarity_score):
    if -0.5 < polarity_score:
        category = 'Mostly Negative'
    elif -0.5 <= polarity_score  < 0:
        category = 'Negative'
    elif polarity_score == 0:
        category = 'Neutral'
    elif 0 < polarity_score <= 0.5:
        category = 'Positive'
    else:
        category = 'Very Positive'
    return category

# genrates an empty dictionary
# returns a dictionary object
def generate_score_dictionary():
    score_dictionary = {
        'positive_score': [],
        'negative_score': [],
        'polarity_score': [],
        'average_sentence_length': [],
        'percentage_of_complex_words': [],
        'fog_index': [],
        'complex_word_count': [],
        'word_count': [],
        'uncertainty_score': [],
        'constraining_score': [],
        'positive_word_proportion': [],
        'negative_word_proportion': [],
        'uncertainty_word_proportion': [],
        'constraining_word_proportion': [],
        
        # extras
        'subjective_score': [],
        'sentence_count': [],
        'category': [],
    }
    
    return score_dictionary


# calculates scores for input file, using word dictionary provided in input and appends it calculated scores to score dictionary
# uses stemming before comparing the file text to word dictionary

def get_scores(file, word_dictionary, score_dictionary):
    text = file.read().lower()
    ps = PorterStemmer()
    
    score_dictionary['sentence_count'].append(len(nltk.tokenize.sent_tokenize(text)))

    words =  get_filtered_text(text, word_dictionary['lm_stopwords'])
    score_dictionary['word_count'].append(len(words))

    all_words = dict(nltk.FreqDist(ps.stem(word) for word in words))
    
    positive_words = word_dictionary['positive_words'].intersection(all_words.keys()) 
    negative_words = word_dictionary['negative_words'].intersection(all_words.keys())
    complex_words = word_dictionary['complex_words'].intersection(all_words.keys())
    uncertainty_words = word_dictionary['uncertainty_words'].intersection(all_words.keys())
    constraining_words = word_dictionary['constraining_words'].intersection(all_words.keys())
    
    # 0 count variables
    score_dictionary['positive_score'].append(sum([all_words[word] for word in positive_words])) 
    score_dictionary['negative_score'].append(sum([all_words[word] for word in negative_words])) 
    score_dictionary['complex_word_count'].append(sum([all_words[word] for word in complex_words])) 
    score_dictionary['uncertainty_score'].append(sum([all_words[word] for word in uncertainty_words])) 
    score_dictionary['constraining_score'].append(sum([all_words[word] for word in constraining_words]))

    # 1 derived variables
    score_dictionary['polarity_score'].append((score_dictionary['positive_score'][-1] - score_dictionary['negative_score'][-1])/((score_dictionary['positive_score'][-1] - score_dictionary['negative_score'][-1]) + 0.000001))

    # 2 readability
    score_dictionary['average_sentence_length'].append( score_dictionary['word_count'][-1] / score_dictionary['sentence_count'][-1] )
    score_dictionary['percentage_of_complex_words'].append( score_dictionary['complex_word_count'][-1] * 100 / score_dictionary['word_count'][-1] )
    score_dictionary['fog_index'].append( 0.4 * (score_dictionary['average_sentence_length'][-1] + score_dictionary['percentage_of_complex_words'][-1]) )

    # 3 proportions
    score_dictionary['positive_word_proportion'].append(score_dictionary['positive_score'][-1] / score_dictionary['word_count'][-1])
    score_dictionary['negative_word_proportion'].append(score_dictionary['negative_score'][-1] / score_dictionary['word_count'][-1])
    score_dictionary['uncertainty_word_proportion'].append(score_dictionary['uncertainty_score'][-1] / score_dictionary['word_count'][-1])
    score_dictionary['constraining_word_proportion'].append(score_dictionary['constraining_score'][-1] / score_dictionary['word_count'][-1])

    # 4 misc These were in the Text Analysis doc but not used as output
    score_dictionary['subjective_score'].append((score_dictionary['positive_score'][-1] + score_dictionary['negative_score'][-1])/(score_dictionary['word_count'][-1]+0.000001))
    score_dictionary['category'].append(categorize_sentiment(score_dictionary['polarity_score'][-1]))

    return (score_dictionary)


# methods to run for main menu choice
def downloadEDGARdata():
        cik_list = pd.read_excel('Dictionaries/cik_list.xlsx')
        sources = list(cik_list['SECFNAME'])
        destination = 'Extracted Data/'
        print('Loaded file url list.')
        
        get_data_from_sec_gov_archives(sources, destination)
        print("Downloaded files are stored in {}".format(destination))
        print("\n----------------------------------------------\n")

def analyzeEDGARdata():
        cik_list = pd.read_excel('Dictionaries/cik_list.xlsx')
        destination = 'Extracted Data/'
        print('\nCreating new dataframe for results.')
        
        results = pd.read_excel('Output/Results.xlsx')
        
        print('Done')
        # copy column data from cik_list to results
        cols = list(cik_list.columns)
        for col in cols:
            results[col] = cik_list[col]
        print('Cik File variables transferred to output dataframe.')
        
        
        word_dict = get_word_dict(positives_file, negatives_file, stopword_file, complex_words_file, uncertainty_words_file, constraining_words_file)
        score_dictionary = generate_score_dictionary()

        doc_list = [doc for doc in os.listdir(destination) if doc.endswith('.txt')]
        total_docs = len(doc_list)
        docs_done = 0
        print('Analyzing, this will take a while.\n')

        printProgressBar(0, total_docs, prefix = 'Progress:', suffix = 'Complete', length = 50)
        
        for doc in doc_list:
            with io.open(destination+doc, 'r', encoding='utf-8') as file:
                score_dictionary = get_scores(file, word_dict, score_dictionary)
            docs_done += 1
            printProgressBar(docs_done, total_docs, prefix = 'Progress:', suffix = 'Complete', length = 50)

        columns = [col for col in list(results.columns) if col not in list(cik_list.columns)]
        for column in columns[:-1]:
            results[column] = score_dictionary[column]
        
        results.loc[0, 'constraining_words_whole_report'] = results['constraining_score'].sum()
        
        print("\n----------------------------------------------\n")
        results.to_excel('Output/Output.xlsx', index=False)
        print('The output has been saved to Output/Output.xlsx')

# Handles execution of all functions
def main():
    
    print('\n\nInitializing Workspace...')
    directories = ['Extracted Data', 'Logs', 'Dictionaries']
    make_directories(directories)
    
    print("\n----------------------------------------------\n")

    while True:
        choice  = int(input('What would you like to do?\
        \n1. Download EDGAR Files (If Extracted Data Folder is empty, you must do this.)\
        \n2. Analyse the EDGAR Data\
        \n3. Exit\n'))
        if(choice == 1):
            print("\n----------------------------------------------\n")
            start_time = time.time()
            if (len(os.listdir('Extracted Data'))== 0):
                downloadEDGARdata()
            else:
                ip = input("There already seem to be files in Extracted Data, you can directly proceed with Analyzing them OR you can choose to redownload the data (this will overwrite the files present).\nPress Y to redownload and N to goto main menu.\n(Y/N)\n")
                if (ip.lower()=='y'):
                    downloadEDGARdata()
                else:
                    os.system('cls')
                    continue
            print("That took {:.2f} minute(s) ".format((time.time() - start_time)/60))
            input('Press any key to continue')
        if(choice == 2):
                os.system('cls')
                start_time = time.time()
                analyzeEDGARdata()
                print("That took {:.2f} minute(s) ".format((time.time() - start_time)/60))
                input('Press any key to continue')
        if(choice == 3):
                print('Quitting')
                time.sleep(1)
                exit()
        else:
                print('Invalid Choice')
                os.system('cls')
    return   
        

if __name__ == "__main__":
    main()




