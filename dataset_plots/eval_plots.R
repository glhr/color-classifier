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
boxplot + geom_boxplot(width=5, outlier.shape = 1) +
  stat_summary(fun=median,
               geom="line",
               aes(group=channels, color=channels),
               position = position_dodge(width = 0.4),
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
ggsave("accuracy-vs-bins.png",width = 10, height=5)

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
tuning_results <- read.csv("~/catkin_ws/src/lh7-nlp/vision_RGB/src/color_classifier/dataset_plots/tuning_results.csv")
data.accuracy <- melt(tuning_results, id.vars=c("X","classifier","dataset"), measure.vars = c("accuracy_default","accuracy_tuned"))
data.timing <- melt(tuning_results, id.vars=c("X","classifier","dataset"), measure.vars = c("time_default","time_tuned"))
tuningplot <- ggplot(data.accuracy,aes(classifier,value))

tuningplot + geom_col(aes(classifier,
                          value,
                          fill=variable,
                          colour=dataset
                          ),
                      position = position_dodge(width = 0.7),
                      width=0.6,
                      size=1) +
  labs(x="Classifier",
       y="Cross-Validation accuracy",
       colour="Colour space\nfor feature vector",
       fill="Classifier\nhyper-parameters",
       sec.axis="CV Computation time (s)") +
  theme(axis.title = element_text(size = 14),
        axis.text = element_text(size = 12),
        legend.title = element_text(size = 14),
        legend.text = element_text(size = 11)) +
  stat_summary(fun=mean,
               geom="point",
               data=data.timing,
             aes(classifier,
                 value/7000,
                 group=variable),
             position=position_dodge(width=0.7),
             size =2.5) +
  #scale_color_manual("mean", values=c("tomato", "green", "green")) + # Will show mean on top of the line
  scale_y_continuous(labels = function(x) paste0(x, ""),
                     limits=c(0, 1),
                     breaks=seq(0,1,0.2),
                     sec.axis = sec_axis(trans= ~.*7,
                                         name = "Cross-Validation computation time (s)")) +
  scale_fill_manual(
    values = c("accuracy_default" = "azure3",
               "accuracy_tuned" = "thistle4"),
    breaks=c("accuracy_default", "accuracy_tuned"),
    labels=c("Default params", "Best params\n(from Grid Search)"),
    aesthetics = c("fill")
  ) +
  scale_colour_manual(
    values = c("dataset-hsv-32-.json" = "darkorange",
               "dataset-ycbcr-32-.json" = "darkblue"),
    breaks=c("dataset-hsv-32-.json", "dataset-ycbcr-32-.json"),
    labels=c("HSV", "YCbCr"),
    aesthetics = c("colour")
  ) +
  geom_point(aes(classifier,
                 value,
                 group=variable,
                 alpha="")) +
  scale_alpha_discrete(name="",
                     labels=c("Average\ncomputation time")) +
  guides(col = guide_legend(override.aes = list(shape = 15, size = 10, fill="white")),
         alpha = guide_legend(override.aes = list(size = 5, colour="black", alpha=1)))
ggsave("accuracy-vs-tuning.pdf",width = 14, height=6)

