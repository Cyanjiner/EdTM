library(readr)
library(patchwork) # To display 2 charts together
library(hrbrthemes)
LDAall_ouput <- read_csv("Desktop/github/EdTM/LDA/LDAall_ouput.csv")
names(LDAall_ouput)[2] <- "Topic_Diversity"
library(ggplot2)
# Most basic line chart
diversity_plot <- ggplot(LDAall_ouput , aes(x=Epoch, y=Topic_Diversity)) +
  geom_line(color="#69b3a2", size=2) +
  ggtitle("Topic Diversity Scores") +
  theme_ipsum()

perplexity_plot <- ggplot(LDAall_ouput , aes(x=Epoch, y=Perplexity)) +
  geom_line(color="grey",size=2) +
  ggtitle("Perplexity Scores") +
  theme_ipsum()

# Display both charts side by side thanks to the patchwork package
diversity_plot + perplexity_plot
