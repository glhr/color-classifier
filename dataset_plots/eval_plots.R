eval_results <- read.csv("~/catkin_ws/src/lh7-nlp/vision_RGB/src/color_classifier/dataset_plots/eval_results.csv")
eval <- subset(eval_results, (histo_eq == 'False'))

# reshape data
library(reshape)
library(ggplot2)

eval_SDG <- subset(eval, (classifier =='SGDClassifier' & histo_eq == 'False'))
eval_BNB <- subset(eval, (classifier =='BernoulliNB' & histo_eq == 'False'))
eval_MNB <- subset(eval, (classifier =='MultinomialNB' & histo_eq == 'False'))

eval$histo_bins = as.factor(eval$histo_bins)
data.accuracy <- melt(eval, id.vars=c("X","histo_bins","channels"), measure.vars = c("accuracy"))
boxplot <- ggplot(data.accuracy,aes(histo_bins, value))
boxplot + geom_boxplot(aes(fill=channels)) +
  stat_summary(fun=median,
               geom="line",
               aes(group=channels, color=channels),
               position = position_dodge(width = 0.75),
               show.legend = FALSE)  + 
  stat_summary(fun=median,
               geom="point",
               aes(group=channels, color=channels),
               position = position_dodge(width = 0.75),
               show.legend = FALSE)  + 
  labs(x="Number of histogram bins",
       y="Cross-Validation accuracy\nacross all classifiers",
       fill="Histogram\ncolour space") +
  scale_fill_discrete(breaks=c("hsv", "rgb", "ycbcr"),
                      labels=c("HSV", "RGB", "YCbCr"))

eval <- subset(eval_results, (histo_eq == 'False'))
data.time <- melt(eval, id.vars=c("X","histo_bins","classifier"), measure.vars = c("time"))
lineplot <- ggplot(data.time, aes(histo_bins, value, color=classifier))
lineplot +
  # geom_point(size=2) +
  stat_summary(fun=mean,
               geom="point",
               aes(group=classifier, color=classifier),
               show.legend = TRUE) +
  stat_summary(fun=mean,
               geom="line",
               aes(group=classifier, color=classifier),
               show.legend = FALSE)
