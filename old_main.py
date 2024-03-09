import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

import warnings

# Ignorar futuros avisos
warnings.simplefilter(action='ignore', category=FutureWarning)

# Ler o arquivo CSV
hotel_reviews = pd.read_csv("Hotel_Reviews.csv")

# Remover as linhas com valores ausentes
hotel_reviews.dropna(inplace=True)

# Exibir informações básicas sobre o dataset
print("\nBasic Information about the dataset:")
print(hotel_reviews.info())

# Exibir o dataset original
print("Original Dataset:")
print(hotel_reviews)

# Exibir as primeiras linhas do dataset
print("\nFirst few rows of the dataset:")
print(hotel_reviews.head())

# Estatísticas resumidas
print("\nSummary Statistics:")
print(hotel_reviews.describe())

# Selecionar as colunas com valor numerico
colunas_numericas = hotel_reviews.select_dtypes(include=['number'])

# Guardar as colunas selecionadas em um novo arquivo CSV
#colunas_numericas.to_csv('colunas_numericas.csv', index=False)

arr = colunas_numericas.iloc[:, 2:].to_numpy()

print("\nSummary Statistics with NumPy:")

#Mean
print("\n*  Mean  *")
print("Average_Score:",np.round(np.mean(arr[:, 0]),2))
print("Review_Total_Negative_Word_Counts:",np.round(np.mean(arr[:, 1]),2))
print("Total_Number_of_Reviews:",np.round(np.mean(arr[:, 2]),2))
print("Review_Total_Positive_Word_Counts:",np.round(np.mean(arr[:, 3]),2))
print("Total_Number_of_Reviews_Reviewer_Has_Given:",np.round(np.mean(arr[:, 4]),2))
print("Reviewer_Score:",np.round(np.mean(arr[:, 5]),2))
print("lat:",np.round(np.mean(arr[:, 6]),2))
print("lng:",np.round(np.mean(arr[:, 7]),2))

# Median
print("\n*  Median  *")
print("Average_Score:",np.round(np.median(arr[:, 0]),2))
print("Review_Total_Negative_Word_Counts:",np.round(np.median(arr[:, 1]),2))
print("Total_Number_of_Reviews:",np.round(np.median(arr[:, 2]),2))
print("Review_Total_Positive_Word_Counts:",np.round(np.median(arr[:, 3]),2))
print("Total_Number_of_Reviews_Reviewer_Has_Given:",np.round(np.median(arr[:, 4]),2))
print("Reviewer_Score:",np.round(np.median(arr[:, 5]),2))
print("lat:",np.round(np.median(arr[:, 6]),2))
print("lng:",np.round(np.median(arr[:, 7]),2))

# Standard Deviation
print("\n*  Standard Deviation  *")
print("Average_Score:",np.round(np.std(arr[:, 0]),2))
print("Review_Total_Negative_Word_Counts:",np.round(np.std(arr[:, 1]),2))
print("Total_Number_of_Reviews:",np.round(np.std(arr[:, 2]),2))
print("Review_Total_Positive_Word_Counts:",np.round(np.std(arr[:, 3]),2))
print("Total_Number_of_Reviews_Reviewer_Has_Given:",np.round(np.std(arr[:, 4]),2))
print("Reviewer_Score:",np.round(np.std(arr[:, 5]),2))
print("lat:",np.round(np.std(arr[:, 6]),2))
print("lng:",np.round(np.std(arr[:, 7]),2))


# Histograma entre o Num de Reviews e o Score
plt.figure(figsize=(9, 7))
plt.hist(hotel_reviews["Reviewer_Score"])
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
sns.displot(hotel_reviews["Reviewer_Score"], kind="kde")
plt.xlim(2, 10)
plt.xlabel("Score")
plt.ylabel("Numb of Reviews")
plt.title("Kernel Density - Score Distribution")
plt.savefig('score_distribution_kde.png', dpi=300, bbox_inches='tight')
plt.show()

# Scatterplot - Relação entre número de reviews feitas e pontuação dada ao hotel
plt.figure(figsize=(9, 7))
sns.scatterplot(x="Total_Number_of_Reviews_Reviewer_Has_Given", y="Reviewer_Score", data=hotel_reviews, alpha=0.5, s=30)
plt.xlabel("Number of Reviews made")
plt.ylabel("Score")
plt.title("Relationship between number of reviews made and score given to the hotel")
plt.savefig('Scatterplot.png', dpi=300, bbox_inches='tight')
plt.show()

# Calcular a média de pontuação para cada país
pontuacao_media_por_pais = hotel_reviews.groupby('Country_Name')['Reviewer_Score'].mean()

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
pessoas_por_pais = hotel_reviews['Country_Name'].value_counts()

# Criar o gráfico de pizza
plt.figure(figsize=(10, 6))
plt.pie(pessoas_por_pais, labels=pessoas_por_pais.index, autopct='%1.1f%%', startangle=140)
plt.title('Amount of data per country')
plt.axis('equal')
plt.savefig('amount_data_per_country.png', dpi=300, bbox_inches='tight')
plt.show()

# Criar o heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(hotel_reviews.pivot_table(index='Country_Name', columns='Reviewer_Score', values='Average_Score'), cmap='coolwarm')
plt.xlabel('Reviewer_Score')
plt.ylabel('Country_Name')
plt.title('Heatmap da Pontuação Média do Hotel por País e Pontuação do Revisor')
plt.show()

# Criar o boxplot
plt.figure(figsize=(12, 12))
sns.boxplot(x='Country_Name', y='Reviewer_Score', data=hotel_reviews)
plt.xlabel('Country_Name')
plt.ylabel('Reviewer_Score')
plt.title('Boxplot da Pontuação do Revisor por País')
plt.xticks(rotation=45)
plt.show()

# Criar o violin plot
plt.figure(figsize=(7, 5))
sns.violinplot(x='Country_Name', y='Reviewer_Score', data=hotel_reviews)
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
sns.heatmap(hotel_reviews.pivot_table(index='Country_Name', columns='Reviewer_Score', values='Average_Score'),
            cmap='coolwarm')
plt.xlabel('Review_Score')
plt.ylabel('Country')
plt.title('Average Score Heatmap by Country')
plt.savefig('average_score_heatmap_country.png', dpi=300, bbox_inches='tight')
plt.show()

# Boxplot do review score por país
plt.figure(figsize=(12, 12))
sns.boxplot(x='Country_Name', y='Reviewer_Score', data=hotel_reviews)
plt.xlabel('Country')
plt.ylabel('Reviewer_Score')
plt.title('Boxplot of Reviewer Score by Country')
plt.xticks(rotation=45)
plt.savefig('Boxplot_Reviewer_Score_Country.png', dpi=300, bbox_inches='tight')
plt.show()

# violin plot - Distribuição da pontuação da review por país
plt.figure(figsize=(7, 5))
sns.violinplot(x='Country_Name', y='Reviewer_Score', data=hotel_reviews)
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
    count_per_range[i] = hotel_reviews[(hotel_reviews['Total_Number_of_Reviews_Reviewer_Has_Given'] >= start) &
                                       (hotel_reviews['Total_Number_of_Reviews_Reviewer_Has_Given'] <= end)].shape[0]

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


'''
colunas_selecionadas = hotel_reviews[['Average_Score','Review_Total_Negative_Word_Counts','Total_Number_of_Reviews','Review_Total_Positive_Word_Counts','Country_Name']]
sns.pairplot(colunas_selecionadas, hue='Country_Name')
plt.savefig('pairplot_all_features1.png', dpi=300, bbox_inches='tight')
plt.show()

colunas_selecionadas2 = hotel_reviews[['Total_Number_of_Reviews_Reviewer_Has_Given','Reviewer_Score','lat','lng','Country_Name']]
sns.pairplot(colunas_selecionadas2, hue='Country_Name')
plt.savefig('pairplot_all_features2.png', dpi=300, bbox_inches='tight')
plt.show()


colunas_selecionadas3 = hotel_reviews[['Average_Score','Review_Total_Negative_Word_Counts','Total_Number_of_Reviews','Review_Total_Positive_Word_Counts','Total_Number_of_Reviews_Reviewer_Has_Given','Reviewer_Score','lat','lng']]

# Criar o heatmap
plt.figure(figsize=(14, 14))
sns.heatmap(colunas_selecionadas3.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Heatmap de Correlação entre Features')
plt.savefig('heatmap_all_features.png', dpi=300, bbox_inches='tight')
plt.show()


colunas_selecionadas4 = hotel_reviews[['Average_Score','Review_Total_Negative_Word_Counts','Total_Number_of_Reviews','Review_Total_Positive_Word_Counts','Country_Name']]
plt.figure(figsize=(14, 14))
sns.violinplot(data=colunas_selecionadas, inner="points", palette="coolwarm")
plt.title('Violin Plot')
plt.savefig('Violin_Plot_all_features.png', dpi=300, bbox_inches='tight')
plt.show()
'''

#%%
'''
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud


hotel_reviews = pd.read_csv("hotel_reviews.csv")

negative_reviews_text = ' '.join(hotel_reviews['Negative_Review'])

words = negative_reviews_text.split()

# Criar dicionário de palavras negativas
negative_words = [
    "awful", "bad", "terrible", "dreadful", "horrible", "disappointed",
    "unhappy", "frustrated", "angry", "irritated", "sad", "disgusting",
    "dirty", "unacceptable", "uncomfortable", "noisy", "crowded", "broken",
    "old", "outdated", "inefficient", "rude", "unhelpful", "disrespectful",
    "expensive", "overpriced", "disappointed", "waste of money", "unfair",
    "deceptive", "dishonest", "misleading", "poor", "inadequate", "insufficient",
    "painful", "unhealthy", "dangerous", "unsafe", "isolated", "remote",
    "boring", "unexciting", "disappointing"
]

# Filtrar palavras negativas
filtered_words = [word for word in words if word in negative_words]

word_counts = Counter(filtered_words)

most_common_word, most_common_count = word_counts.most_common(1)[0]

plt.figure(figsize=(10, 6))
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_counts)
plt.imshow(wordcloud, interpolation='bilinear')
plt.title(f'Most Common Negative Word: "{most_common_word}" (Count: {most_common_count})')
plt.axis('off')
plt.show()
'''
#%%
'''
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud


hotel_reviews = pd.read_csv("hotel_reviews.csv")

negative_reviews_text = ' '.join(hotel_reviews['Positive_Review'])

words = negative_reviews_text.split()

# Criar dicionário de palavras negativas
positive_words = [
    "wonderful", "good", "fantastic", "pleasant", "splendid",
    "delighted", "joyful", "content", "calm", "serene",
    "cheerful", "appealing", "clean", "acceptable", "peaceful",
    "spacious", "functional", "new", "modern", "efficient",
    "polite", "supportive", "respectful", "affordable", "reasonable",
    "satisfied", "worthwhile", "just", "honest", "truthful", "accurate",
    "rich", "ample", "sufficient", "pleasurable", "wholesome", "secure",
    "connected", "thriving", "engrossing", "stimulating", "exciting",
    "fulfilling"
]


# Filtrar palavras negativas
filtered_words = [word for word in words if word in positive_words]

word_counts = Counter(filtered_words)

most_common_word, most_common_count = word_counts.most_common(1)[0]

plt.figure(figsize=(10, 6))
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_counts)
plt.imshow(wordcloud, interpolation='bilinear')
plt.title(f'Most Common Negative Word: "{most_common_word}" (Count: {most_common_count})')
plt.axis('off')
plt.show()
'''

# plt.figure(figsize=(20, 20))
# #sns.heatmap(hotel_reviews.pivot_table(index='lat', columns='lng', values='Average_Score',), cmap='coolwarm')
# sns.scatterplot(x='lng', y='lat', size='Average_Score', hue='Average_Score',
#                 sizes=(50, 200), palette='viridis', data=hotel_reviews)
#
# plt.title("Ubicação do hotel")
# plt.xlabel("Longitude")
# plt.ylabel("Latitude")
# plt.grid(True)
# plt.show()

#%% Data Pre-processing

import pandas as pd
import numpy as np

# Ignorar futuros avisos
warnings.simplefilter(action='ignore', category=FutureWarning)

# Ler o arquivo CSV
hotel_reviews = pd.read_csv("Hotel_Reviews.csv")

# Display the original dataset
print("Original Dataset:")
print(hotel_reviews)

# Dealing with missing values
# Option 1: Delete missing values
dropped_df = hotel_reviews.dropna(inplace=True)

# Option 2: Interpolate missing values (only valid if done withing samples of the same class to be tested, for example, island)
dropped_interpolate_df = hotel_reviews.interpolate()

# Display datasets after handling missing values
print("With dropna: ", dropped_df)
print("With interpolate: ", dropped_interpolate_df)

# Dealing with outliers for all numerical columns
print("\nDetecting outliers:")
for feature in hotel_reviews.select_dtypes(include=[np.number]).columns:
    # Calculate the IQR (InterQuartile Range)
    Q1 = hotel_reviews[feature].quantile(0.25)
    Q3 = hotel_reviews[feature].quantile(0.75)
    IQR = Q3 - Q1

    # Define the lower and upper bounds to identify outliers
    upper_bound = Q3 + 1.5 * IQR
    lower_bound = Q1 - 1.5 * IQR

    outliers = hotel_reviews[(hotel_reviews[feature] < lower_bound) | (hotel_reviews[feature] > upper_bound)]
    # Each row represents a data point where the values for the features are listed.
    # The first row presents the indices of the DataFrame where the outliers were detected.
    print(f"Outliers in '{feature}':\n{outliers}" if not outliers.empty else f"No outliers in '{feature}'.")

# Display dataset after handling outliers for all numerical columns
print(hotel_reviews)

# Impute missing values using the mean value of the column
#penguins_imputed = hotel_reviews.fillna(penguins.iloc[:,2:6].mean())

# Display dataset after imputing missing values

# %% Examine dimensionality reduction algorithms

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import warnings

import umap


class DimensionalityReduction:
    def __init__(self, data, targets):
        """
        Initialize the DimensionalityReduction object with the dataset.

        Parameters:
        - data: The dataset to perform dimensionality reduction on.
        - targets: The targets of the samples.
        """
        self.test = data
        self.data = StandardScaler().fit_transform(data)
        self.targets = targets

    def compute_pca(self, n_components=2):
        """
        Compute Principal Component Analysis (PCA) on the dataset.

        Parameters:
        - n_components: The number of components to keep.

        Returns:
        - pca_projection: The projected data using PCA.
        """
        return PCA(n_components=n_components).fit_transform(self.data)

    def compute_lda(self, n_components=2):
        """
        Perform Linear Discriminant Analysis (LDA) on the input data.

        Parameters:
        - n_components: The number of components to keep

        Returns:
            array-like: The reduced-dimensional representation of the data using LDA.
        """
        return LinearDiscriminantAnalysis(n_components=n_components).fit_transform(self.data, self.targets)
    def compute_tsne(self, n_components=2, perplexity=3):

        """
        Compute t-Distributed Stochastic Neighbor Embedding (t-SNE) on the dataset.

        Parameters:
        - n_components: The number of components to embed the data into.
        - perplexity: The perplexity parameter for t-SNE.

        Returns:
        - tsne_projection: The projected data using t-SNE.
        """
        return TSNE(n_components=n_components, perplexity=perplexity).fit_transform(self.data)

    def compute_umap(self, n_components=2, n_neighbors=8, min_dist=0.5, metric='euclidean'):
        """
        Compute Uniform Manifold Approximation and Projection (UMAP) on the dataset.

        Parameters:
        - n_components: The number of components to embed the data into.
        - n_neighbors: The number of neighbors to consider for each point.
        - min_dist: The minimum distance between embedded points.
        - metric: The distance metric to use.

        Returns:
        - umap_projection: The projected data using UMAP.
        """
        return umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist,
                         metric=metric).fit_transform(self.data)

    def compute_lle(self, n_components=2, n_neighbors=20):
        """
        Compute Locally Linear Embedding (LLE) on the dataset.

        Parameters:
        - n_components: The number of components to embed the data into.
        - n_neighbors: The number of neighbors to consider for each point.

        Returns:
        - lle_projection: The projected data using LLE.
        """
        return LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=n_components).fit_transform(self.data)

    def plot_projection(self, projection, title):
        """
        Plot the 2D projection of the dataset.

        Parameters:
        - projection: The projected data.
        - title: The title of the plot.
        """
        plt.figure(figsize=(8, 6))
        plt.scatter(projection[:, 0], projection[:, 1], c=self.targets, alpha=0.5)
        plt.title(title)
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    # Ignorar futuros avisos
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # Ler o arquivo CSV

    chunk_size = 1000
    hotel_reviews = pd.read_csv("Hotel_Reviews.csv")

    # Remover as linhas com valores ausentes
    hotel_reviews.dropna(inplace=True)

    #Bins
    bins_reviewer_score = [0, 4, 7, 10]
    hotel_reviews['Reviewer_Score_bin'] = pd.cut(hotel_reviews['Reviewer_Score'], bins=bins_reviewer_score,
                                                 labels=['Mau', 'Neutral', 'Bom'])

    # Create an instance of LabelEncoder
    label_encoder = LabelEncoder()

    # Encode the 'species' column
    hotel_reviews['Reviewer_Score_bin_encoded'] = label_encoder.fit_transform(hotel_reviews['Reviewer_Score_bin'])

    # Initialize DimensionalityReduction object with the dataset
    dr = DimensionalityReduction(hotel_reviews[['Average_Score', 'Review_Total_Negative_Word_Counts',
                                                'Review_Total_Negative_Word_Counts', 'Total_Number_of_Reviews',
                                                'Review_Total_Positive_Word_Counts',
                                                'Total_Number_of_Reviews_Reviewer_Has_Given', 'lat', 'lng']],
                                 hotel_reviews['Reviewer_Score_bin_encoded'])

    # Compute and plot PCA projection
    print('PCA')
    dr.plot_projection(dr.compute_pca(), 'PCA Projection')
    # Compute and plot LDA projection
    print('LDA')
    dr.plot_projection(dr.compute_lda(), 'LDA Projection')
    # Compute and plot t-SNE projection
    print('t-SNE')
    dr.plot_projection(dr.compute_tsne(), 't-SNE Projection')
    # Compute and plot UMAP projection
    print('UMAP')
    dr.plot_projection(dr.compute_umap(), 'UMAP Projection')
    # Compute and plot LLE projection
    print('LLE')
    dr.plot_projection(dr.compute_lle(), 'LLE Projection')


# %% Feature Engineering - Binning and Interaction Features

import pandas as pd
import numpy as np
import warnings


# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', None)

def count_tags(tags):
    array_tags = tags.split(',')
    return len(array_tags)


# Ignorar futuros avisos
warnings.simplefilter(action='ignore', category=FutureWarning)

# Ler o arquivo CSV
hotel_reviews = pd.read_csv("Hotel_Reviews.csv")

# Remover as linhas com valores ausentes
hotel_reviews.dropna(inplace=True)

# Display the initial dataset
print('Initial Dataset: \n', hotel_reviews)

# Choose the 'sepal length (cm)' column for binning
feature_to_bin = 'Reviewer_Score'
feature_to_bin2 = 'Total_Number_of_Reviews_Reviewer_Has_Given'
feature_to_bin3 = 'Total_Number_of_Reviews'

# Define the number of bins (or bin edges)
bins_reviewer_score = [0, 4, 7,
                       10]  # Perguntar ao prof se é melhor ter mais bins: 0-4 Mau, 4-6 Neutral, 6-8 Bom, 8-10 Muito Bom
bins_expertise_level = [0, 5, 10, 15, 20, float('inf')]
bins_total_reviews = [0, 2500, 5000, 7500, float('inf')]\

# Perform binning using pandas
hotel_reviews['Reviewer_Score_bin'] = pd.cut(hotel_reviews[feature_to_bin], bins=bins_reviewer_score,
                                             labels=['Mau', 'Neutral', 'Bom'])
hotel_reviews['Reviewer_Expertise_Level_bin'] = pd.cut(hotel_reviews[feature_to_bin2], bins=bins_expertise_level,
                                                       labels=['Nenhum', 'Baixo', 'Medio', 'Alto', 'Experto'])
hotel_reviews['Review_Count_Per_Hotel_bin'] = pd.cut(hotel_reviews[feature_to_bin3], bins=bins_total_reviews,
                                                     labels=['Poucas_Avaliações', 'Algumas_Avaliações',
                                                             'Muitas_Avaliações', 'Muitissimas_Avaliações'])

# Display the dataset after binning
print('Dataset after binning score reviewer: \n', hotel_reviews[['Reviewer_Score', 'Reviewer_Score_bin']])
print('Dataset after binning expertise level: \n',
      hotel_reviews[['Total_Number_of_Reviews_Reviewer_Has_Given', 'Reviewer_Expertise_Level_bin']].head(20))
print('Dataset after binning expertise level: \n',
      hotel_reviews[['Total_Number_of_Reviews', 'Review_Count_Per_Hotel_bin']])

# New Features
# Simple Combinations

mean_words_positive = hotel_reviews['Review_Total_Positive_Word_Counts'].mean()
print('Mean of Positive Words: ', mean_words_positive)
mean_words_negative = hotel_reviews['Review_Total_Negative_Word_Counts'].mean()
print('Mean of Negative Words: ', mean_words_negative)

hotel_reviews['Ratio_WordCount_Between_Positive_Negative_Reviews'] = \
    (hotel_reviews['Review_Total_Positive_Word_Counts'] / hotel_reviews['Review_Total_Negative_Word_Counts'])

hotel_reviews['Deviation_Between_AverageScore_ReviewerScore'] = (
        hotel_reviews['Reviewer_Score'] - hotel_reviews['Average_Score'])

hotel_reviews['Ratio_Between_AverageScore_TotalReviews'] = \
    (hotel_reviews['Total_Number_of_Reviews'] / hotel_reviews['Average_Score'])

hotel_reviews['Total_Tags'] = hotel_reviews['Tags'].apply(count_tags)

hotel_reviews['Total_Review_Words'] = hotel_reviews['Review_Total_Positive_Word_Counts'] + hotel_reviews[
    'Review_Total_Negative_Word_Counts']

hotel_reviews['Ratio_Total_Positive_Words'] = hotel_reviews['Review_Total_Positive_Word_Counts'] / hotel_reviews[
    'Total_Review_Words']

hotel_reviews['Ratio_Positive_Words_Mean'] = hotel_reviews['Review_Total_Positive_Word_Counts'] / mean_words_positive

hotel_reviews['Ratio_Total_Negative_Words'] = hotel_reviews['Review_Total_Negative_Word_Counts'] / hotel_reviews[
    'Total_Review_Words']

hotel_reviews['Ratio_Negative_Words_Mean'] = hotel_reviews['Review_Total_Negative_Word_Counts'] / mean_words_negative

hotel_reviews['Reviewer_Has_Given_Positive_Review'] = hotel_reviews['Review_Total_Positive_Word_Counts'].apply(
    lambda review: 0 if review == 0 else 1)

hotel_reviews['Reviewer_Has_Given_Negative_Review'] = hotel_reviews['Review_Total_Negative_Word_Counts'].apply(
    lambda review: 0 if review == 0 else 1)

hotel_reviews['Log_Reviewer_Score'] = np.log(hotel_reviews['Reviewer_Score'])

hotel_reviews['Square_Ratio_WordCount'] = np.power(hotel_reviews['Ratio_WordCount_Between_Positive_Negative_Reviews'], 2)

hotel_reviews['Weighted_Average_Reviewer_Score'] = ((hotel_reviews['Reviewer_Score'] *
                                                    hotel_reviews['Total_Number_of_Reviews_Reviewer_Has_Given']) +
                                                    hotel_reviews['Log_Reviewer_Score'])

# Display the dataset after creating interactions
print('Dataset after creating interactions: \n', hotel_reviews.head(20))
