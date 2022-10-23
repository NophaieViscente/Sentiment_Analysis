import pandas as pd
from ProcessingData import ProcessData
import seaborn as sns
from wordcloud import WordCloud
import matplotlib.pyplot as plt


class PlotData :

    def __init__(self, processor_data=ProcessData()) -> None:
        self.processor = processor_data

    def plot_target_balance(
        self, 
        data:pd.DataFrame,
        column_target:str,
        dict_map_classes:dict)->sns:
        
        count = data[column_target].value_counts()
        sns.barplot(x=count.index.map(dict_map_classes), y=count)
        plt.gca().set_ylabel('samples')
        plt.title('Count per Target')

        return plt.show()
    
    def plot_num_char_per_text(
        self, data:pd.DataFrame,
        column_text:str)->plt :

        dataframe = data.copy()
        dataframe['num_chars'] = self.processor.count_chars(data=dataframe[column_text])
        fig = plt.figure(figsize=(10,5))
        sns.histplot(data=dataframe, x='num_chars', hue='target', element='step')
        plt.title("Number of Characters")
        plt.show()
    
    def plot_number_of_words(
        self, data:pd.DataFrame,
        column_text:str)->plt :

        dataframe = data.copy()
        dataframe['num_words'] = self.processor.count_number_words(data=dataframe[column_text])
        fig = plt.figure(figsize=(15, 5))
        sns.histplot(data=dataframe, x="num_words", hue='target')
        plt.title("Number of Words")
        plt.show()
    
    def plot_wordcloud(
        self, data:pd.DataFrame,
        column_text:str,
        column_target:str,
        target:int=0,
        background_color:str='white',
        width:int=800,height:int=400)->WordCloud :

        wc = WordCloud(
            background_color=background_color,
            max_font_size=80, max_words=150, 
            width=width, height=height).generate(
                " ".join(self.processor.create_corpus(
                    data=data, target=target, column_target=column_target,
                    column_text=column_text, specific_target=False)))
        plt.imshow(wc)
        plt.axis('off')
        plt.show()
        
    def plot_count_sentences(
        self, data:pd.DataFrame,
        column_sentence:str,
        display_most_appear:int=20)->sns:

        chains= data[column_sentence].value_counts()[:display_most_appear]
        sns.barplot(x=chains,y=chains.index,palette='deep')
        plt.title(f"Top {display_most_appear} Keywords")
        plt.xlabel("Count of Keywords")
    
    def plot_most_common_terms(
        self, data:pd.DataFrame,
        column_with_terms:str,
        column_with_target:str,
        qtd_to_plot:int=20) :

        corpus = self.processor.create_corpus(
            data=data, target=0, 
            column_text=column_with_terms,
            column_target=column_with_target,
            specific_target=False)
        most_common_words, counts = self.processor.count_most_common_terms(corpus)

        most_common_words = most_common_words[:qtd_to_plot]
        counts = counts[:qtd_to_plot]
        fig = plt.figure(figsize=(15,8))
        sns.barplot(x= counts, y= most_common_words)
        plt.xticks(rotation=75)
        plt.title(f"{qtd_to_plot} Most Common Terms")
        plt.show()

