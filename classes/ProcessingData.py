import pandas as pd
import re
from collections import defaultdict, Counter
from nltk.corpus import stopwords
import string

class ProcessData :

    '''
        This class process the data
    '''
    def __init__(self, language_stopwords:str='english') -> None:
        self.stopwords = set(stopwords.words(language_stopwords))
        pass

    def count_number_words(
        self, data:pd.Series)->pd.Series:
        '''
            This method count number of words, of each sentence
            in a Pandas Series.

            Args:
                data: A Series with data to count number of words
        '''
        qtd_words = data.apply(lambda x: len(x.split()))

        return qtd_words

    def count_chars(
        self, data:pd.Series)->pd.Series :

        '''
            This method count the number of characters, of each 
            sentence in a Pandas Series.

            Args:
                data: A Series with data to count number of chars
        '''
        qtd_chars = data.str.len()

        return qtd_chars
    
    def create_corpus(
        self, data:pd.DataFrame, 
        target:int, column_text:str, 
        column_target:str, specific_target:bool=True)->list :

        '''
            This method create a corpus with all texts from dataframe.

            Args:
                data: A dataframe with data to create corpus
                target: A number of target to looking for
                column_text: A column with text data
                column_target: A column with data target
                specific_target: A boolean to determine, if looking
                to whole data or a specific target number
        '''
        corpus = list()
        if specific_target :
            
            for list_ in data[data[column_target] == target][column_text].str.split() :
                for word in list_ :
                    corpus.append(word)
            return corpus
        else :
            
            for list_ in data[column_text].str.split() :
                for word in list_ :
                    corpus.append(word)
            return corpus
                


    def stopwords_per_target(
        self, corpus:list)->dict :

        dic = defaultdict()
        for word in corpus :
            if word in self.stopwords :
                dic[word]+=1
        
        return dic
    
    def remove_URLs(
        self, data:pd.Series)->pd.Series:

        url_chars = re.compile(r'https?://\S+|www\.\S+')
        data_cleaned = data.apply(lambda x: url_chars.sub(r'',x))

        return data_cleaned

    def remove_HTMLs (
        self, data:pd.Series)->pd.Series :

        html_tags = re.compile(r'<.*?>')
        data_cleaned = data.apply(lambda txt: html_tags.sub(r'',txt))

        return data_cleaned
    
    def clean_non_ascii_chars(
        self, data:pd.Series)->pd.Series:

        data_cleaned = data.apply(lambda txt: ''.join([x for x in txt if x in string.printable]))

        return data_cleaned

    def remove_emojis(
        self, data:pd.Series)->pd.Series:

        emoji_codes = re.compile(
            "["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE)
        data_cleaned = data.apply(lambda txt: emoji_codes.sub(r'',txt))

        return data_cleaned
    
    def remove_mention(
        self, data:pd.Series)->pd.Series:

        mention = re.compile(r'@\S+')
        data_cleaned = data.apply(lambda txt: mention.sub(r'', txt))

        return data_cleaned
    
    def remove_numbers(
        self, data:pd.Series)->pd.Series:
        
        number = re.compile(r'[-+]?[.\d]*[\d]+[:,.\d]*')
        data_cleaned = data.apply(lambda txt: number.sub(r'', txt))

        return data_cleaned
    

    def remove_punctuation(
        self, data:pd.Series,
        punctuation:str=string.punctuation.replace('"',''))->pd.Series:

        punctuation = re.compile(fr"[{punctuation}]")
        data_cleaned = data.apply(lambda x: punctuation.sub(r'',x))

        return data_cleaned
    
    def remove_stopwords(
        self, data:pd.Series):

        data_cleaned = data.apply(lambda txt: ' '.join([x for x in txt.split() if x not in self.stopwords]))
        return data_cleaned
    
    def clean_whole_text(
        self, data:pd.DataFrame,
        column_to_clean:str)->pd.DataFrame :

        dataframe = data.copy()
        text = self.remove_URLs(dataframe[column_to_clean])
        text = self.remove_HTMLs(text)
        text = self.clean_non_ascii_chars(text)
        text = self.remove_emojis(text)
        text = self.remove_mention(text)
        text = self.remove_stopwords(text)
        text = self.remove_numbers(text)
        text = self.remove_punctuation(text)
        text = self.remove_stopwords(text)
        dataframe[column_to_clean] = text

        return dataframe
    
    def count_most_common_terms(
        self, corpus:list) :

        counter = Counter(corpus)
        most_common = counter.most_common()

        words = list()
        counts = list()
        for word, count in most_common :
            words.append(word)
            counts.append(count)
        
        return words, counts

