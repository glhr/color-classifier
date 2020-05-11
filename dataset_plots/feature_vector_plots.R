
library(gridExtra)
library(grid)
library(ggplot2)
library(magick)



plot_feature_vector <- function(y, y.norm, channels, image) {
  y.max = max(unlist(y))
  n_bins = length(y)/3
  x = seq(0, 255, length.out=n_bins)
  x.norm = seq(0, n_bins*3-1)
  feature_vector.1 = do.call(rbind, Map(data.frame, x=x, y=y[0:n_bins]))
  feature_vector.2 = do.call(rbind, Map(data.frame, x=x, y=y[(n_bins+1):(n_bins*2)]))
  feature_vector.3 = do.call(rbind, Map(data.frame, x=x, y=y[(n_bins*2+1):(n_bins*3)]))
  feature_vector.norm = do.call(rbind, Map(data.frame, x=x.norm, y=y.norm))

  n_ticks = length(seq(0, y.max, 2000))
  labels = replicate(n_ticks, "")
  
  raster <- as.raster(image)
  g <- rasterGrob(image, width=1, height=unit(1,"npc"), interpolate = FALSE)
  plot.img <- ggplot(feature_vector.1, aes(x = x, y = y)) + coord_fixed() +
    annotation_custom(g, -Inf, Inf, -Inf, Inf) +
    theme(axis.title.x=element_blank(),
          axis.text.x=element_blank(),
          axis.ticks.x=element_blank(),
          axis.title.y=element_blank(),
          axis.text.y=element_blank(),
          axis.ticks.y=element_blank())

  
  plot.1 <- ggplot(feature_vector.1, aes(x = x, y = y)) + geom_col() +
    scale_x_discrete(limits=c(0, 255)) +
    scale_y_continuous(breaks=seq(0, y.max, 2000),
                       limits=c(0, y.max)) +
    labs(x=element_blank(),
         y=element_blank(),
         title=paste(channels[1], "channel")) +
    theme_minimal() +
    theme(plot.title = element_text(size = 10, hjust = 0.5))
  plot.2 <- ggplot(feature_vector.2, aes(x = x, y = y)) + geom_col() +
    scale_x_discrete(limits=c(0, 255)) +
    scale_y_continuous(breaks=seq(0, y.max, 2000),
                       limits=c(0, y.max),
                       labels = labels) +
    labs(x=element_blank(),
         y=element_blank(),
         title=paste(channels[2], "channel")) +
    theme_minimal() +
    theme(plot.title = element_text(size = 10, hjust = 0.5),
          axis.ticks = element_blank())
  plot.3 <- ggplot(feature_vector.3, aes(x = x, y = y)) + geom_col() +
    scale_x_discrete(limits=c(0, 255)) +
    scale_y_continuous(breaks=seq(0, y.max, 2000),
                       limits=c(0, y.max),
                       labels = labels) +
    labs(title=paste(channels[3], "channel"),
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

  plot.row = arrangeGrob(plot.1, plot.2, plot.3, ncol=3,
                      left = textGrob("Bin counts", rot = 90, vjust = 0.5, ),
                      top = textGrob("Histogram for each channel of the masked image",
                                     gp = gpar(fontsize = 12, fontface='italic')),
                      bottom = textGrob("Intensity value", vjust=-0.5))
  return(arrangeGrob(plot.img, plot.row, plot.norm, 
                     layout_matrix = rbind(c(NA, 1, NA),
                                           c(2, 2, 2),
                                           c(2, 2, 2),
                                           c(3, 3, 3),
                                           c(3, 3, 3))))
}

image <- image_read('5.png')

# HSV 16 bins, object 1
y = list(186, 9319, 2608, 8, 8, 25, 3, 0, 12, 24, 138, 17, 10, 11, 2, 0, 2081, 1252, 258, 688, 1280, 3824, 2913, 75, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 105, 11, 2909, 458, 494, 1058, 1086, 1680, 1852, 2003, 657, 51, 0, 0)
y.norm = list(0.015350621032202054, 0.7690991258015643, 0.2152388153332417, 0.0006602417648258948, 0.0006602417648258948, 0.0020632555150809214, 0.00024759066180971056, 0.0, 0.0009903626472388422, 0.0019807252944776844, 0.011389170443246685, 0.0014030137502550266, 0.0008253022060323686, 0.0009078324266356053, 0.0001650604412064737, 0.0, 0.17174538907533587, 0.10332783619525254, 0.02129279691563511, 0.05678079177502695, 0.10563868237214318, 0.3155955635867777, 0.24041053261722894, 0.0061897665452427635, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.000577711544222658, 0.00866567316333987, 0.0009078324266356053, 0.240080411734816, 0.03779884103628248, 0.040769928977999005, 0.08731697339822458, 0.08962781957511522, 0.13865077061343792, 0.15284596855719465, 0.1653080318682834, 0.054222354936326614, 0.004209041250765079, 0.0, 0.0)
channels = c("Hue", "Value", "Saturation")
plot.hsv = plot_feature_vector(y, y.norm, channels, image)
ggsave("featurevector_hsv_5.png", plot.hsv, width = 20, height = 20, units = "cm")

# RGB 16 bins, object 1
y = list(0, 7, 105, 11, 2909, 458, 494, 1058, 1086, 1680, 1855, 2017, 649, 42, 0, 0, 0, 16, 102, 12, 3042, 646, 1025, 1301, 1819, 2002, 1730, 469, 187, 20, 0, 0, 0, 80, 49, 1119, 3025, 2884, 2655, 1570, 344, 269, 143, 82, 133, 18, 0, 0)
y.norm = list(0.0, 0.0008167762718969572, 0.012251644078454357, 0.0012835055701237899, 0.33942888213546407, 0.05344050464697234, 0.05764106833101383, 0.12344989938099724, 0.12671700446858505, 0.1960263052552697, 0.21644571205269364, 0.23534824863088036, 0.0757268286373036, 0.004900657631381743, 0.0, 0.0, 0.0, 0.0018669171929073306, 0.011901597104784233, 0.0014001878946804979, 0.35494763130150625, 0.07537678166363347, 0.11959938267062586, 0.15180370424827733, 0.21224514836865216, 0.23359801376252975, 0.20186042148310512, 0.05472401021709613, 0.021819594692104427, 0.0023336464911341633, 0.0, 0.0, 0.0, 0.009334585964536653, 0.0057174339032787, 0.13056752117895642, 0.3529640317840422, 0.33651182402154634, 0.30979157169806015, 0.18319124955403182, 0.040138719647507606, 0.031387545305754494, 0.016685572411609267, 0.009567950613650069, 0.015518749166042185, 0.002100281842020747, 0.0, 0.0)
channels = c("Red", "Green", "Blue")
plot.rgb = plot_feature_vector(y, y.norm, channels, image)
ggsave("featurevector_rgb_5.png", plot.rgb, width = 20, height = 20, units = "cm")


# YCBCR 16 bins, object 1
y = list(0, 0, 32, 90, 2848, 713, 1098, 1581, 2137, 2473, 979, 376, 44, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6135, 5687, 549, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 31, 12317, 23, 0, 0, 0, 0, 0, 0)
y.norm = list(0.0, 0.0, 0.002039791193777696, 0.005736912732499771, 0.18154141624621498, 0.0454490975363593, 0.0699903353364972, 0.10077843366757931, 0.13621980565946679, 0.15763761319413258, 0.062404861834636394, 0.023967546526887932, 0.0028047128914443324, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3910662179320677, 0.36250914121918, 0.0349951676682486, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0019760477189721434, 0.7851283791799964, 0.0014660999205277192, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
channels = c("Y", "Cb", "Cr")
plot.ycbcr = plot_feature_vector(y, y.norm, channels, image)
ggsave("featurevector_ycbcr_5.png", plot.ycbcr, width = 20, height = 20, units = "cm")
