#from typing import re

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


class PreVisualization:
    def __init__(self, data_loader):
        self.data_loader = data_loader

    def perform_pre_visualization(self):
        self.numpy_statistics()

        self.create_plots()

    def numpy_statistics(self):
        # Select numerical columns
        numerical_columns = self.data_loader.dataframe.select_dtypes(include=['number'])

        arr = numerical_columns.to_numpy()

        print("\nSummary Statistics with NumPy:")

        # Mean
        print("\n*  Mean  *")
        print("Average_Score:", np.round(np.mean(arr[:, 0]), 2))
        print("Review_Total_Negative_Word_Counts:", np.round(np.mean(arr[:, 1]), 2))
        print("Total_Number_of_Reviews:", np.round(np.mean(arr[:, 2]), 2))
        print("Review_Total_Positive_Word_Counts:", np.round(np.mean(arr[:, 3]), 2))
        print("Total_Number_of_Reviews_Reviewer_Has_Given:", np.round(np.mean(arr[:, 4]), 2))
        print("Reviewer_Score:", np.round(np.mean(arr[:, 5]), 2))
        print("lat:", np.round(np.mean(arr[:, 6]), 2))
        print("lng:", np.round(np.mean(arr[:, 7]), 2))

        # Median
        print("\n*  Median  *")
        print("Average_Score:", np.round(np.median(arr[:, 0]), 2))
        print("Review_Total_Negative_Word_Counts:", np.round(np.median(arr[:, 1]), 2))
        print("Total_Number_of_Reviews:", np.round(np.median(arr[:, 2]), 2))
        print("Review_Total_Positive_Word_Counts:", np.round(np.median(arr[:, 3]), 2))
        print("Total_Number_of_Reviews_Reviewer_Has_Given:", np.round(np.median(arr[:, 4]), 2))
        print("Reviewer_Score:", np.round(np.median(arr[:, 5]), 2))
        print("lat:", np.round(np.median(arr[:, 6]), 2))
        print("lng:", np.round(np.median(arr[:, 7]), 2))

        # Standard Deviation
        print("\n*  Standard Deviation  *")
        print("Average_Score:", np.round(np.std(arr[:, 0]), 2))
        print("Review_Total_Negative_Word_Counts:", np.round(np.std(arr[:, 1]), 2))
        print("Total_Number_of_Reviews:", np.round(np.std(arr[:, 2]), 2))
        print("Review_Total_Positive_Word_Counts:", np.round(np.std(arr[:, 3]), 2))
        print("Total_Number_of_Reviews_Reviewer_Has_Given:", np.round(np.std(arr[:, 4]), 2))
        print("Reviewer_Score:", np.round(np.std(arr[:, 5]), 2))
        print("lat:", np.round(np.std(arr[:, 6]), 2))
        print("lng:", np.round(np.std(arr[:, 7]), 2))

    def create_plots(self):

        # Histograma entre o Num de Reviews e o Score
        plt.figure(figsize=(9, 7))
        plt.hist(self.data_loader.dataframe["Reviewer_Score"])
        plt.xlim(2.5, 10)
        plt.xlabel("Score")
        plt.ylabel("Numb of Reviews")
        plt.title("Score Distribution")
        plt.savefig('score_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()

        '''
        plt.figure(figsize=(9, 7), facecolor='black')  # Define o fundo da figura como preto
        plt.hist(hotel_reviews["Reviewer_Score"], color='red')  # Define a cor do histograma como vermelho
        plt.xlim(2, 10)
        plt.xlabel("Score", color='white')  # Define a cor do texto do eixo x como branco
        plt.ylabel("Numb of Reviews", color='white')  # Define a cor do texto do eixo y como branco
        plt.title("Score Distribution", color='white')  # Define a cor do título como branco
        plt.gca().spines['bottom'].set_color('white')  # Define a cor da linha do eixo x como branco
        plt.gca().spines['left'].set_color('white')  # Define a cor da linha do eixo y como branco
        plt.tick_params(axis='x', colors='white')  # Define a cor dos números do eixo x como branco
        plt.tick_params(axis='y', colors='white')  # Define a cor dos números do eixo y como branco
        plt.savefig('score_distribution2.png', dpi=300, bbox_inches='tight')
        plt.show()
        '''

        # Densidade do Kernel entre o Num de Reviews e o Score
        plt.figure(figsize=(9, 9))
        sns.displot(self.data_loader.dataframe["Reviewer_Score"], kind="kde")
        plt.xlim(2, 10)
        plt.xlabel("Score")
        plt.ylabel("Numb of Reviews")
        plt.title("Kernel Density - Score Distribution")
        plt.savefig('score_distribution_kde.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Scatterplot - Relação entre número de reviews feitas e pontuação dada ao hotel
        plt.figure(figsize=(9, 7))
        sns.scatterplot(x="Total_Number_of_Reviews_Reviewer_Has_Given", y="Reviewer_Score", data=self.data_loader.dataframe,
                        alpha=0.5, s=30)
        plt.xlabel("Number of Reviews made")
        plt.ylabel("Score")
        plt.title("Relationship between number of reviews made and score given to the hotel")
        plt.savefig('Scatterplot.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Calcular a média de pontuação para cada país
        pontuacao_media_por_pais = self.data_loader.dataframe.groupby('Country_Name')['Reviewer_Score'].mean()

        # Criar o gráfico de barras
        plt.figure(figsize=(10, 6))
        pontuacao_media_por_pais.plot(kind='bar', color='skyblue')

        # Adicionar rótulos e título
        plt.xlabel('Country')
        plt.ylabel('Average Hotel Score')
        plt.title('Average Hotel Score by Country')

        # Adicionar as médias como anotações nas barras
        for i, pontuacao in enumerate(pontuacao_media_por_pais):
            plt.text(i, pontuacao, f'{pontuacao:.2f}', ha='center', va='bottom')

        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y')
        plt.tight_layout()
        plt.savefig('average_scores_by_country.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Quantidade Paises representados (Hoteis)
        pessoas_por_pais = self.data_loader.dataframe['Country_Name'].value_counts()

        # Criar o gráfico de pizza
        plt.figure(figsize=(10, 6))
        plt.pie(pessoas_por_pais, labels=pessoas_por_pais.index, autopct='%1.1f%%', startangle=140)
        plt.title('Amount of data per country')
        plt.axis('equal')
        plt.savefig('amount_data_per_country.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Criar o heatmap
        plt.figure(figsize=(12, 6))
        sns.heatmap(
            self.data_loader.dataframe.pivot_table(index='Country_Name', columns='Reviewer_Score', values='Average_Score'),
            cmap='coolwarm')
        plt.xlabel('Reviewer_Score')
        plt.ylabel('Country_Name')
        plt.title('Heatmap da Pontuação Média do Hotel por País e Pontuação do Revisor')
        plt.show()

        # Criar o boxplot
        plt.figure(figsize=(12, 12))
        sns.boxplot(x='Country_Name', y='Reviewer_Score', data=self.data_loader.dataframe)
        plt.xlabel('Country_Name')
        plt.ylabel('Reviewer_Score')
        plt.title('Boxplot da Pontuação do Revisor por País')
        plt.xticks(rotation=45)
        plt.show()

        # Criar o violin plot
        plt.figure(figsize=(7, 5))
        sns.violinplot(x='Country_Name', y='Reviewer_Score', data=self.data_loader.dataframe)
        plt.xlabel('País')
        plt.ylabel('Pontuação do Revisor')
        plt.title('Distribuição da Pontuação do Revisor por País')
        plt.xticks(rotation=45)
        plt.show()

        '''
        # Quantidade Paises representados (Reviewer_Nationality)
        pessoas_por_pais = hotel_reviews['Reviewer_Nationality'].value_counts()
        # Criar o gráfico de pizza
        plt.figure(figsize=(10, 6))
        plt.pie(pessoas_por_pais, labels=pessoas_por_pais.index, autopct='%1.1f%%', startangle=140)
        plt.title('Amount of data per country')
        plt.axis('equal')
        plt.savefig('amount_data_per_country.png', dpi=300, bbox_inches='tight')
        plt.show()
        '''

        # Heatmap da Pontuação Média por País
        plt.figure(figsize=(12, 7))
        sns.heatmap(
            self.data_loader.dataframe.pivot_table(index='Country_Name', columns='Reviewer_Score', values='Average_Score'),
            cmap='coolwarm')
        plt.xlabel('Review_Score')
        plt.ylabel('Country')
        plt.title('Average Score Heatmap by Country')
        plt.savefig('average_score_heatmap_country.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Boxplot do review score por país
        plt.figure(figsize=(12, 12))
        sns.boxplot(x='Country_Name', y='Reviewer_Score', data=self.data_loader.dataframe)
        plt.xlabel('Country')
        plt.ylabel('Reviewer_Score')
        plt.title('Boxplot of Reviewer Score by Country')
        plt.xticks(rotation=45)
        plt.savefig('Boxplot_Reviewer_Score_Country.png', dpi=300, bbox_inches='tight')
        plt.show()

        # violin plot - Distribuição da pontuação da review por país
        plt.figure(figsize=(7, 5))
        sns.violinplot(x='Country_Name', y='Reviewer_Score', data=self.data_loader.dataframe)
        plt.xlabel('Country')
        plt.ylabel('Reviewer_Score')
        plt.title('Reviewer Score Distribution by Country')
        plt.xticks(rotation=45)
        plt.savefig('reviewer_score_distribution_country_vp.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Definir as faixas de número de reviews
        review_ranges = [(0, 3), (4, 6), (7, 9), (10, 12), (13, 15), (16, 18), (19, 21), (21, 24), (25, float('inf'))]

        # Inicializar contadores para cada faixa
        count_per_range = [0] * len(review_ranges)

        # Contar o número de pessoas em cada faixa de reviews
        for i, (start, end) in enumerate(review_ranges):
            count_per_range[i] = \
            self.data_loader.dataframe[(self.data_loader.dataframe['Total_Number_of_Reviews_Reviewer_Has_Given'] >= start) &
                                  (self.data_loader.dataframe[
                                       'Total_Number_of_Reviews_Reviewer_Has_Given'] <= end)].shape[0]

        # Definir rótulos das faixas
        labels = [f'{start}-{end}' if end != float('inf') else f'{start}+' for start, end in review_ranges]

        # Plotar o gráfico de barras
        plt.figure(figsize=(10, 6))
        plt.bar(labels, count_per_range, color='skyblue')
        plt.title('Quantidade de Pessoas por Faixa de Número de Reviews Dados')
        plt.xlabel('Faixa de Número de Reviews')
        plt.ylabel('Quantidade de Pessoas')
        plt.grid(axis='y')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    def plot_positive_negative_reviews(self):
        # Converte a coluna 'Review_Date' para o tipo datetime
        self.data_loader.dataframe['Review_Date'] = pd.to_datetime(self.data_loader.dataframe['Review_Date'])

        # Extrai o mês da data
        self.data_loader.dataframe['Month'] = self.data_loader.dataframe['Review_Date'].dt.month

        # Conta a quantidade de revisões por mês
        reviews_count_by_month = self.data_loader.dataframe.groupby('Month').size()

        # Cria o gráfico de barras
        plt.figure(figsize=(10, 6))
        plt.bar(reviews_count_by_month.index, reviews_count_by_month, color='skyblue')

        # Define rótulos nos eixos e um título
        plt.xlabel('Mês')
        plt.ylabel('Número de Revisões')
        plt.title('Número de Revisões por Mês')

        # Ajusta os ticks do eixo X e os limites do eixo X
        plt.xticks(range(1, 13),
                   ['Janeiro', 'Fevereiro', 'Março', 'Abril', 'Maio', 'Junho', 'Julho', 'Agosto', 'Setembro', 'Outubro',
                    'Novembro', 'Dezembro'], rotation=45)
        plt.xlim(0.5, 12.5)  # Ajustar os limites do eixo X para que todas as barras sejam visíveis

        plt.grid(axis='y')
        plt.tight_layout()
        plt.show()

        # Calcula a média de palavras positivas para cada pontuação de revisão
        avg_positive_words = self.data_loader.dataframe.groupby('Reviewer_Score')['Review_Total_Positive_Word_Counts'].mean()

        # Cria o gráfico de barras
        plt.figure(figsize=(10, 6))
        avg_positive_words.plot(kind='bar', color='lightgreen')
        plt.xlabel('Pontuação do Revisor')
        plt.ylabel('Contagem Média de Palavras Positivas')
        plt.title('Contagem Média de Palavras Positivas por Pontuação do Revisor')
        plt.grid(axis='y')
        plt.tight_layout()
        plt.show()

        # Calcula a média de palavras negativas para cada pontuação de revisão
        avg_negative_words = self.data_loader.dataframe.groupby('Reviewer_Score')['Review_Total_Negative_Word_Counts'].mean()

        # Cria o gráfico de barras
        plt.figure(figsize=(10, 6))
        avg_negative_words.plot(kind='bar', color='salmon', alpha=0.7)
        plt.xlabel('Pontuação do Revisor')
        plt.ylabel('Contagem Média de Palavras Negativas')
        plt.title('Contagem Média de Palavras Negativas por Pontuação do Revisor')
        plt.grid(axis='y')
        plt.tight_layout()
        plt.show()

    def process_reviews(self, reviews_column):
        # Combina as reviews num unico texto
        all_reviews = ' '.join(reviews_column)

        # Tokeniza o texto em palavras, ignorando numeros e Case Sensitive
        #words = re.findall(r'\b[A-Za-z]+\b', all_reviews.lower())

        # Filtra palavras comuns usando NLTK
        #english_stopwords = set(stopwords.words('english'))
        #filtered_words = [word for word in words if word not in english_stopwords]

        # Conta a frequência de cada palavra
        #word_counts = Counter(filtered_words)

        # Obtem as 100 palavras mais comuns
        #top_100_words = word_counts.most_common(100)

        # Converte a lista de tuplos num dicionario
        #word_freq_dict = dict(top_100_words)

        # Cria um DataFrame para mostrar todas as palavras e a sua frequência
        #df = pd.DataFrame(top_100_words, columns=['Palavra', 'Nº de Repetições'])

        #return word_freq_dict, df

    def show_wordcloud(self, word_freq, title=None):
        #wordcloud = WordCloud(
            background_color='white',
            max_words=200,
            max_font_size=40,
            scale=3,
            random_state=42
        #)

        #wordcloud.generate_from_frequencies(word_freq)

        #plt.figure(figsize=(10, 6))
        #plt.imshow(wordcloud, interpolation="bilinear")
        #plt.axis('off')
        #if title:
        #    plt.title(title, fontsize=20)
        #plt.show()

    def create_nice_wordcloud(self):

        # Filtra apenas as palavras de reviews positivas
        positive_reviews = self.data_loader.dataframe[self.data_loader.dataframe['Positive_Review'] != 'No Positive']
        word_freq_dict_pos, df_pos = self.process_reviews(positive_reviews['Positive_Review'])

        # Mostra a tabela de palavras mais repetidas en reviews positivas
        print("\nTabela de palavras mais repetidas em Reviews Positivas:\n")
        print(df_pos)

        # Mostra a nube de palavras nas reviews positivas
        self.show_wordcloud(word_freq_dict_pos, "Nuvem de Palavras Mais Comuns em Reviews Positivas")

        # Filtra apenas as palavras de reviews negativas
        negative_reviews = self.data_loader.dataframe[self.data_loader.dataframe['Negative_Review'] != 'No Negative']
        word_freq_dict_neg, df_neg = self.process_reviews(negative_reviews['Negative_Review'])

        # Mostra a tabela de palavras mais repetidas en reviews negativas
        print("\nTabela de palavras mais repetidas em Reviews Negativas:\n")
        print(df_neg)

        # Mostra a nube de palavras nas reviews negativas
        self.show_wordcloud(word_freq_dict_neg, "Nuvem de Palavras Mais Comuns em Reviews Negativas")
