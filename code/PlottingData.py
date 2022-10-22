import pandas as pd
from ProcessingData import ProcessData
import seaborn as sns
from wordcloud import WordCloud
import matplotlib.pyplot as plt


class PlotData :

    def __init__(self, processor_data=ProcessData()) -> None:
        self.processor = processor_data

    def plotting_target_balance(
        self, data:pd.DataFrame,
        column_target:str)->sns:

        count = data[column_target].value_counts()
        sns.barplot(x=count.index, y=count)
        plt.gca().set_ylabel('samples')
        plt.title('Count Target')
        return plt.show()
    
    def plotting_num_char_per_text(
        self, data:pd.DataFrame,
        column_text:str)->plt :

        dataframe = data.copy()
        dataframe['num_chars'] = self.processor.count_chars(data=dataframe[column_text])
        fig = plt.figure(figsize=(10,5))
        sns.histplot(data=dataframe, x='num_chars', hue='target')
        plt.title("Number of Characters")
        plt.show()
    
    def plotting_number_of_words(
        self, data:pd.DataFrame,
        column_text:str)->plt :

        dataframe = data.copy()
        dataframe['num_words'] = self.processor.count_number_words(data=dataframe[column_text])
        fig = plt.figure(figsize=(15, 5))
        sns.histplot(data=dataframe, x="num_words")
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