eval_results <- read.csv("~/catkin_ws/src/lh7-nlp/vision_RGB/src/color_classifier/dataset_plots/eval_results.csv")
eval <- subset(eval_results, (histo_eq == 'False'))

# reshape data
library(reshape)
library(ggplot2)

eval_SDG <- subset(eval, (classifier =='SGDClassifier' & histo_eq == 'False'))
eval_BNB <- subset(eval, (classifier =='BernoulliNB' & histo_eq == 'False'))
eval_MNB <- subset(eval, (classifier =='MultinomialNB' & histo_eq == 'False'))

# eval$histo_bins = as.factor(eval$histo_bins)
data.accuracy <- melt(eval, id.vars=c("X","histo_bins","channels"), measure.vars = c("accuracy"))
boxplot <- ggplot(data.accuracy,aes(histo_bins, value,
                                    group=interaction(histo_bins, channels),
                                    color=channels))
boxplot + geom_boxplot(width=2, outlier.shape = 1) +
  stat_summary(fun=median,
               geom="line",
               aes(group=channels, color=channels),
               position = position_dodge(width = 0.2),
               show.legend = FALSE)  + 
  theme(axis.title = element_text(size = 14),
        axis.text = element_text(size = 12),
        legend.text = element_text(size = 12)) +
  # stat_summary(fun=median,
  #              geom="point",
  #              aes(group=channels, color=channels),
  #              position = position_dodge(width = 0),
  #              show.legend = FALSE)  + 
  labs(x="Number of histogram bins",
       y="Cross-Validation accuracy\nacross all classifiers",
       color="Histogram\ncolour space") +
  scale_color_discrete(breaks=c("hsv", "rgb", "ycbcr"),
                      labels=c("HSV", "RGB", "YCbCr"))
ggsave("accuracy-vs-bins.png",width = 10, height=6)

eval <- subset(eval_results, (histo_eq == 'False'))
eval$time = eval$time/1000
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
               show.legend = FALSE) +
  labs(x="Number of histogram bins",
       y="Cross-Validation computation time (160 samples)\nin seconds",
       color="Classifier") +
  theme(axis.title = element_text(size = 14),
        axis.text = element_text(size = 12),
        legend.text = element_text(size = 10))
ggsave("time-vs-bins.png",width = 10, height=6)

## TUNING PLOTS
tuning_results <- read.csv("~/catkin_ws/src/lh7-nlp/vision_RGB/src/color_classifier/dataset_plots/tuning_results_robotpc.csv")
data.accuracy <- melt(tuning_results, id.vars=c("X","classifier","dataset"), measure.vars = c("accuracy_default","accuracy_tuned"))
data.timing <- melt(tuning_results, id.vars=c("X","classifier","dataset"), measure.vars = c("time_default","time_tuned"))
tuningplot <- ggplot(data.accuracy,aes(classifier,value))
tuningplot + geom_col(aes(classifier,
                          value,
                          alpha=variable,
                          fill=dataset),
                      position = "dodge",
                      width=0.7) +
  # geom_text(aes(x=classifier,
  #               y=value,
  #               group=dataset,
  #               label=round(value,3)),
  #           position=position_dodge(width=0.7),
  #           vjust=+1.5) +
  labs(x="Classifier",
       y="Cross-Validation accuracy",
       alpha="Hyper-parameters",
       sec.axis="CV Computation time (s)") +
  scale_alpha_discrete(breaks=c("accuracy_default", "accuracy_tuned"),
                       labels=c("Default params", "Best params\n(from Grid Search)")) +
  theme(axis.title = element_text(size = 14),
        axis.text = element_text(size = 12),
        legend.title = element_text(size = 14),
        legend.text = element_text(size = 11)) +
  stat_summary(fun=mean,
               geom="point",
               data=data.timing,
             aes(classifier,
                 value/3000,
                 group=variable),
             position=position_dodge(width=0.9),
             size = 2) +
  scale_alpha_discrete() +
  scale_y_continuous(labels = function(x) paste0(x, ""),
                     limits=c(0, 1),
                     breaks=seq(0,1,0.2),
                     sec.axis = sec_axis(trans= ~.*3,
                                         name = "Cross-Validation computation time (s)"))
ggsave("accuracy-vs-tuning.png",width = 14, height=6)

