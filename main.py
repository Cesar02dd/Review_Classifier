import pandas as pd
import seaborn as sns
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
plt.figure(figsize=(10,6))
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
sns.heatmap(hotel_reviews.pivot_table(index='Country_Name', columns='Reviewer_Score', values='Average_Score'), cmap='coolwarm')
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
review_ranges = [(0,3),(4, 6), (7, 9), (10, 12), (13, 15), (16,18),(19,21), (21,24), (25, float('inf'))]

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


