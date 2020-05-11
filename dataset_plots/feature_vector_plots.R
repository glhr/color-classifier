
library(gridExtra)
library(grid)
library(ggplot2)



plot_feature_vector <- function(y, y.norm, channels) {
  y.max = max(unlist(y))
  n_bins = length(y)/3
  x = seq(0, 255, length.out=n_bins)
  x.norm = seq(0, n_bins*3-1)
  feature_vector.1 = do.call(rbind, Map(data.frame, x=x, y=y[0:n_bins]))
  feature_vector.2 = do.call(rbind, Map(data.frame, x=x, y=y[(n_bins+1):(n_bins*2)]))
  feature_vector.3 = do.call(rbind, Map(data.frame, x=x, y=y[(n_bins*2+1):(n_bins*3)]))
  feature_vector.norm = do.call(rbind, Map(data.frame, x=x.norm, y=y.norm))
  
  plot.1 <- ggplot(feature_vector.1, aes(x = x, y = y)) + geom_col() +
    scale_x_discrete(limits=c(0, 255)) +
    scale_y_continuous(breaks=seq(0, y.max, 2000),
                       limits=c(0, y.max)) +
    labs(x=element_blank(),
         y=element_blank(),
         title=paste(channels[3], " channel")) +
    theme_minimal() +
    theme(plot.title = element_text(size = 10, hjust = 0.5))
  plot.2 <- ggplot(feature_vector.2, aes(x = x, y = y)) + geom_col() +
    scale_x_discrete(limits=c(0, 255)) +
    scale_y_continuous(breaks=seq(0, y.max, 2000),
                       limits=c(0, y.max),
                       labels = c("","","","","")) +
    labs(x=element_blank(),
         y=element_blank(),
         title=paste(channels[3], " channel")) +
    theme_minimal() +
    theme(plot.title = element_text(size = 10, hjust = 0.5),
          axis.ticks = element_blank())
  plot.3 <- ggplot(feature_vector.3, aes(x = x, y = y)) + geom_col() + 
    scale_x_discrete(limits=c(0, 255)) +
    scale_y_continuous(breaks=seq(0, y.max, 2000),
                       limits=c(0, y.max),
                       labels = c("","","","","")) +
    labs(title=paste(channels[3], " channel"),
         y=element_blank(),
         x=element_blank()) +
    theme_minimal() +
    theme(plot.title = element_text(size = 10, hjust = 0.5))
  plot.norm <- ggplot(feature_vector.norm, aes(x = x, y = y)) + geom_col() + 
    scale_x_discrete(limits=seq(0,n_bins*3-1,n_bins),
                     breaks=seq(0,n_bins*3-1,n_bins)) +
    labs(x="Vector index",
         y="Vector value",
         title="Feature vector (normalized to unit length)") +
    theme(plot.title = element_text(size = 12, face = 'italic', hjust = 0.5))
  
  plot.row = arrangeGrob(plot.1, plot.2, plot.3, 
                      ncol=3,
                      left = textGrob("Bin counts", rot = 90, vjust = 0.5, ),
                      top = textGrob("Histogram for each channel of the masked image",
                                     gp = gpar(fontsize = 12, fontface='italic')),
                      bottom = textGrob("Intensity value", vjust=-0.5))
  return(arrangeGrob(plot.row, plot.norm, nrow=2))
}

# HSV 16 bins, object 9
y = list(186, 9319, 2608, 8, 8, 25, 3, 0, 12, 24, 138, 17, 10, 11, 2, 0, 2081, 1252, 258, 688, 1280, 3824, 2913, 75, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 105, 11, 2909, 458, 494, 1058, 1086, 1680, 1852, 2003, 657, 51, 0, 0)
y.norm = list(0.015350621032202054, 0.7690991258015643, 0.2152388153332417, 0.0006602417648258948, 0.0006602417648258948, 0.0020632555150809214, 0.00024759066180971056, 0.0, 0.0009903626472388422, 0.0019807252944776844, 0.011389170443246685, 0.0014030137502550266, 0.0008253022060323686, 0.0009078324266356053, 0.0001650604412064737, 0.0, 0.17174538907533587, 0.10332783619525254, 0.02129279691563511, 0.05678079177502695, 0.10563868237214318, 0.3155955635867777, 0.24041053261722894, 0.0061897665452427635, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.000577711544222658, 0.00866567316333987, 0.0009078324266356053, 0.240080411734816, 0.03779884103628248, 0.040769928977999005, 0.08731697339822458, 0.08962781957511522, 0.13865077061343792, 0.15284596855719465, 0.1653080318682834, 0.054222354936326614, 0.004209041250765079, 0.0, 0.0)
channels = c("Hue", "Value", "Saturation")
plot.hsv = plot_feature_vector(y, y.norm, channels)
ggsave("featurevector_hsv.png", plot.hsv, width = 20, height = 15, units = "cm")

