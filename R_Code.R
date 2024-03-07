
statistics  <- function() {
  # Read the data into a data frame
  colunas_numericas <- read.csv("colunas_numericas.csv")
  
  # Select columns except the first one
  arr <- colunas_numericas[, 2:ncol(colunas_numericas)]
  
  # Print summary statistics
  cat("\nSummary Statistics with R:\n")
  
  # Mean
  cat("\n*  Mean  *\n")
  cat("Average_Score:", round(mean(arr[, 1]), 2), "\n")
  cat("Review_Total_Negative_Word_Counts:", round(mean(arr[, 2]), 2), "\n")
  cat("Total_Number_of_Reviews:", round(mean(arr[, 3]), 2), "\n")
  cat("Review_Total_Positive_Word_Counts:", round(mean(arr[, 4]), 2), "\n")
  cat("Total_Number_of_Reviews_Reviewer_Has_Given:", round(mean(arr[, 5]), 2), "\n")
  cat("Reviewer_Score:", round(mean(arr[, 6]), 2), "\n")
  cat("lat:", round(mean(arr[, 7]), 2), "\n")
  cat("lng:", round(mean(arr[, 8]), 2), "\n")
  
  # Median
  cat("\n*  Median  *\n")
  cat("Average_Score:", round(median(arr[, 1]), 2), "\n")
  cat("Review_Total_Negative_Word_Counts:", round(median(arr[, 2]), 2), "\n")
  cat("Total_Number_of_Reviews:", round(median(arr[, 3]), 2), "\n")
  cat("Review_Total_Positive_Word_Counts:", round(median(arr[, 4]), 2), "\n")
  cat("Total_Number_of_Reviews_Reviewer_Has_Given:", round(median(arr[, 5]), 2), "\n")
  cat("Reviewer_Score:", round(median(arr[, 6]), 2), "\n")
  cat("lat:", round(median(arr[, 7]), 2), "\n")
  cat("lng:", round(median(arr[, 8]), 2), "\n")
  
  # Standard Deviation
  cat("\n*  Standard Deviation  *\n")
  cat("Average_Score:", round(sd(arr[, 1]), 2), "\n")
  cat("Review_Total_Negative_Word_Counts:", round(sd(arr[, 2]), 2), "\n")
  cat("Total_Number_of_Reviews:", round(sd(arr[, 3]), 2), "\n")
  cat("Review_Total_Positive_Word_Counts:", round(sd(arr[, 4]), 2), "\n")
  cat("Total_Number_of_Reviews_Reviewer_Has_Given:", round(sd(arr[, 5]), 2), "\n")
  cat("Reviewer_Score:", round(sd(arr[, 6]), 2), "\n")
  cat("lat:", round(sd(arr[, 7]), 2), "\n")
  cat("lng:", round(sd(arr[, 8]), 2), "\n")
}

pie_chart  <- function() {
  hotel_reviews <- read.csv("Hotel_Reviews.csv")

  # Count the occurrences of each country
  pessoas_por_pais <- table(hotel_reviews$Country_Name)

  # Create the pie chart
  pie(pessoas_por_pais, labels = names(pessoas_por_pais), col = rainbow(length(pessoas_por_pais)),  
      main = "Amount of data per country", startangle = 140, clock = TRUE, percent = TRUE, showpercent = TRUE)                        
}

heatmap <- function() {
  install.packages("corrplot")

  library(corrplot)

  hotel_reviews <- read.csv("Hotel_Reviews.csv")

  # Selecione as colunas 
  colunas_selecionadas <- hotel_reviews[, c('Average_Score', 'Review_Total_Negative_Word_Counts', 'Total_Number_of_Reviews', 'Review_Total_Positive_Word_Counts', 'Total_Number_of_Reviews_Reviewer_Has_Given', 'Reviewer_Score', 'lat', 'lng')]

  # Calcule a matriz de correlação
  correlation_matrix <- cor(colunas_selecionadas)

  # Crie o grafico
  corrplot(correlation_matrix, method="color", type="upper", addCoef.col="black", tl.col="black", tl.srt=45)

  # Adicionar título
  title("Heatmap de Correlação entre Features")

  ## Guardar o plot
  #png("heatmap_all_features.png", width=800, height=800, res=300)
  #corrplot(correlation_matrix, method="color", type="upper", addCoef.col="black", tl.col="black", tl.srt=45)
  #title("Heatmap de Correlação entre Features")
  #dev.off()
}

violin <- function() {
  #install.packages("ggplot2")
  library(ggplot2) 

  hotel_reviews <- read.csv("Hotel_Reviews.csv")

  # Create the violin plot
  ggplot(hotel_reviews, aes(x = Country_Name, y = Reviewer_Score)) +
    geom_violin() +
    labs(x = 'Country', y = 'Reviewer Score', title = 'Reviewer Score Distribution by Country') +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))

  # Save the plot
  ggsave('reviewer_score_distribution_country_vp.png', plot = violin(), dpi = 300, width = 7, height = 5)
}

violin()
#statistics()
#pie_chart()
#heatmap()
