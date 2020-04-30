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
  labs(x="Number of histogram bins",
       y="Cross-Validation accuracy\nacross all classifiers",
       fill="Color space\nfor generating\nhistogram") +
  scale_fill_discrete(breaks=c("hsv", "rgb", "ycbcr"),
                      labels=c("HSV", "RGB", "YCbCr")) +
  stat_summary(fun=mean, geom="line", aes(group="channels"))  + 
    stat_summary(fun=mean, geom="point")

