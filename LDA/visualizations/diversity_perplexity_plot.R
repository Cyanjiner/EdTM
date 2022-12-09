library(readr)
LDAall_ouput <- read_csv("LDA/LDAall_ouput.csv")
names(LDAall_ouput)[2] <- "Topic_Diversity"
library(ggplot2)
# Most basic line chart
diversity_plot <- ggplot(LDAall_ouput , aes(x=Epoch, y=Topic_Diversity)) +
  theme_classic() +
  theme(plot.title = element_text(hjust=0.5))+
  geom_line(color="#0072B2") +
  xlab("Epoch")+
  ylab("Topic Diversity")+
  ggtitle("LDA: Diversity increases during training") 

perplexity_plot <- ggplot(LDAall_ouput , aes(x=Epoch, y=Perplexity)) +
  theme_classic() +
  theme(plot.title = element_text(hjust=0.5))+
  geom_line(color="#0072B2") +
  xlab("Epoch")+
  ylab("Perplexity")+
  ggtitle("LDA: Perplexity decreases during training") 

diversity_plot 
perplexity_plot
