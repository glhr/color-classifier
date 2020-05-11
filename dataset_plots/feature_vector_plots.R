
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

  round_to <- function(x, to = 10) round(x/to)*to
  ticks = round_to(y.max/4, 1000)
  n_ticks = length(seq(0, y.max, ticks))
  labels = replicate(n_ticks, "")
  
  raster <- as.raster(image)
  w = dim(raster)[1]
  h = dim(raster)[2]
  w = 1*(h/w)
  print(w)
  g <- rasterGrob(image, width=unit(w*2.5,"cm"), height=unit(2.5,"cm"), interpolate = FALSE)
  plot.img <- ggplot(feature_vector.1, aes(x = x, y = y)) +
    annotation_custom(g, -Inf, Inf, -Inf, Inf) +
      theme_minimal() + 
    theme(axis.title.x=element_blank(),
          axis.text.x=element_blank(),
          axis.ticks.x=element_blank(),
          axis.title.y=element_blank(),
          axis.text.y=element_blank(),
          axis.ticks.y=element_blank())

  
  plot.1 <- ggplot(feature_vector.1, aes(x = x, y = y)) + geom_col() +
    scale_x_discrete(limits=c(0, 255)) +
    scale_y_continuous(breaks=seq(0, y.max, ticks),
                       limits=c(0, y.max)) +
    labs(x=element_blank(),
         y=element_blank(),
         title=paste(channels[1], "channel")) +
    theme_minimal() +
    theme(plot.title = element_text(size = 10, hjust = 0.5))
  plot.2 <- ggplot(feature_vector.2, aes(x = x, y = y)) + geom_col() +
    scale_x_discrete(limits=c(0, 255)) +
    scale_y_continuous(breaks=seq(0, y.max, ticks),
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
    scale_y_continuous(breaks=seq(0, y.max, ticks),
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
                     layout_matrix = rbind(c(1, 1, 1),
                                           c(2, 2, 2),
                                           c(2, 2, 2),
                                           c(3, 3, 3),
                                           c(3, 3, 3))))
}

PLOT_SIZE = 15

image <- image_read('5.png')

# HSV 16 bins, object 1
y = list(186, 9319, 2608, 8, 8, 25, 3, 0, 12, 24, 138, 17, 10, 11, 2, 0, 2081, 1252, 258, 688, 1280, 3824, 2913, 75, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 105, 11, 2909, 458, 494, 1058, 1086, 1680, 1852, 2003, 657, 51, 0, 0)
y.norm = list(0.015350621032202054, 0.7690991258015643, 0.2152388153332417, 0.0006602417648258948, 0.0006602417648258948, 0.0020632555150809214, 0.00024759066180971056, 0.0, 0.0009903626472388422, 0.0019807252944776844, 0.011389170443246685, 0.0014030137502550266, 0.0008253022060323686, 0.0009078324266356053, 0.0001650604412064737, 0.0, 0.17174538907533587, 0.10332783619525254, 0.02129279691563511, 0.05678079177502695, 0.10563868237214318, 0.3155955635867777, 0.24041053261722894, 0.0061897665452427635, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.000577711544222658, 0.00866567316333987, 0.0009078324266356053, 0.240080411734816, 0.03779884103628248, 0.040769928977999005, 0.08731697339822458, 0.08962781957511522, 0.13865077061343792, 0.15284596855719465, 0.1653080318682834, 0.054222354936326614, 0.004209041250765079, 0.0, 0.0)
channels = c("Hue", "Value", "Saturation")
plot.hsv = plot_feature_vector(y, y.norm, channels, image)
ggsave("featurevector_hsv_5.png", plot.hsv, width = PLOT_SIZE, height = PLOT_SIZE, units = "cm")

# RGB 16 bins, object 1
y = list(0, 7, 105, 11, 2909, 458, 494, 1058, 1086, 1680, 1855, 2017, 649, 42, 0, 0, 0, 16, 102, 12, 3042, 646, 1025, 1301, 1819, 2002, 1730, 469, 187, 20, 0, 0, 0, 80, 49, 1119, 3025, 2884, 2655, 1570, 344, 269, 143, 82, 133, 18, 0, 0)
y.norm = list(0.0, 0.0008167762718969572, 0.012251644078454357, 0.0012835055701237899, 0.33942888213546407, 0.05344050464697234, 0.05764106833101383, 0.12344989938099724, 0.12671700446858505, 0.1960263052552697, 0.21644571205269364, 0.23534824863088036, 0.0757268286373036, 0.004900657631381743, 0.0, 0.0, 0.0, 0.0018669171929073306, 0.011901597104784233, 0.0014001878946804979, 0.35494763130150625, 0.07537678166363347, 0.11959938267062586, 0.15180370424827733, 0.21224514836865216, 0.23359801376252975, 0.20186042148310512, 0.05472401021709613, 0.021819594692104427, 0.0023336464911341633, 0.0, 0.0, 0.0, 0.009334585964536653, 0.0057174339032787, 0.13056752117895642, 0.3529640317840422, 0.33651182402154634, 0.30979157169806015, 0.18319124955403182, 0.040138719647507606, 0.031387545305754494, 0.016685572411609267, 0.009567950613650069, 0.015518749166042185, 0.002100281842020747, 0.0, 0.0)
channels = c("Red", "Green", "Blue")
plot.rgb = plot_feature_vector(y, y.norm, channels, image)
ggsave("featurevector_rgb_5.png", plot.rgb, width = PLOT_SIZE, height = PLOT_SIZE, units = "cm")


# YCBCR 16 bins, object 1
y = list(0, 0, 32, 90, 2848, 713, 1098, 1581, 2137, 2473, 979, 376, 44, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6135, 5687, 549, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 31, 12317, 23, 0, 0, 0, 0, 0, 0)
y.norm = list(0.0, 0.0, 0.002039791193777696, 0.005736912732499771, 0.18154141624621498, 0.0454490975363593, 0.0699903353364972, 0.10077843366757931, 0.13621980565946679, 0.15763761319413258, 0.062404861834636394, 0.023967546526887932, 0.0028047128914443324, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3910662179320677, 0.36250914121918, 0.0349951676682486, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0019760477189721434, 0.7851283791799964, 0.0014660999205277192, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
channels = c("Y", "Cb", "Cr")
plot.ycbcr = plot_feature_vector(y, y.norm, channels, image)
ggsave("featurevector_ycbcr_5.png", plot.ycbcr, width = PLOT_SIZE, height = PLOT_SIZE, units = "cm")


image <- image_read('1.png')

# HSV 16 bins, object 1
y = list(11, 1, 28206, 0, 3, 6, 0, 0, 8, 55, 44, 0, 0, 0, 0, 0, 279, 106, 80, 70, 71, 57, 158, 196, 184, 173, 306, 190, 1597, 13566, 10842, 459, 0, 0, 219, 470, 20, 45, 188, 516, 1182, 3495, 11031, 9722, 1309, 136, 1, 0)
y.norm = list(0.0003013514438725115, 2.7395585806591954e-05, 0.7727198932607326, 0.0, 8.218675741977585e-05, 0.0001643735148395517, 0.0, 0.0, 0.00021916468645273563, 0.0015067572193625575, 0.001205405775490046, 0.0, 0.0, 0.0, 0.0, 0.0, 0.007643368440039155, 0.002903932095498747, 0.002191646864527356, 0.0019176910064614367, 0.0019450865922680286, 0.0015615483909757412, 0.004328502557441528, 0.0053695348180920225, 0.00504078778841292, 0.004739436344540408, 0.008383049256817137, 0.005205161303252471, 0.043750750533127346, 0.37164851705222646, 0.29702294131506995, 0.012574573885225706, 0.0, 0.0, 0.005999633291643638, 0.012875925329098217, 0.000547911716131839, 0.001232801361296638, 0.005150370131639287, 0.014136122276201448, 0.03238158242339169, 0.09574757239403887, 0.30220070703251584, 0.26633988521168694, 0.035860821820828864, 0.0037257996696965054, 2.7395585806591954e-05, 0.0)
channels = c("Hue", "Value", "Saturation")
plot.hsv = plot_feature_vector(y, y.norm, channels, image)
ggsave("featurevector_hsv.png", plot.hsv, width = PLOT_SIZE, height = PLOT_SIZE, units = "cm")

# RGB 16 bins, object 1
y = list(0, 0, 219, 470, 20, 45, 188, 516, 1182, 3495, 11031, 9724, 1344, 99, 1, 0, 0, 0, 518, 179, 30, 188, 566, 1570, 5673, 11108, 7860, 247, 315, 79, 1, 0, 6904, 17353, 2954, 209, 109, 75, 81, 88, 71, 76, 70, 102, 152, 90, 0, 0)
y.norm = list(0.0, 0.0, 0.007692361303791618, 0.016508720606310777, 0.000702498749204714, 0.0015806221857106065, 0.006603488242524311, 0.01812446772948162, 0.0415176760779986, 0.12276165642352377, 0.38746318512385997, 0.34155489186333193, 0.04720791594655678, 0.003477368808563334, 3.51249374602357e-05, 0.0, 0.0, 0.0, 0.01819471760440209, 0.0062873638053821896, 0.001053748123807071, 0.006603488242524311, 0.019880714602493403, 0.05514615181257004, 0.1992637702119171, 0.39016780530829814, 0.27608200843745256, 0.008675859552678217, 0.011064355299974245, 0.00277487005935862, 3.51249374602357e-05, 0.0, 0.24250256822546726, 0.60952303974747, 0.10375906525753625, 0.007341111929189261, 0.003828618183165691, 0.002634370309517677, 0.0028451199342790913, 0.0030909944965007412, 0.0024938705596767347, 0.002669495246977913, 0.002458745622216499, 0.003582743620944041, 0.005338990493955826, 0.003161244371421213, 0.0, 0.0)
channels = c("Red", "Green", "Blue")
plot.rgb = plot_feature_vector(y, y.norm, channels, image)
ggsave("featurevector_rgb.png", plot.rgb, width = PLOT_SIZE, height = PLOT_SIZE, units = "cm")


# YCBCR 16 bins, object 1
y = list(0, 0, 0, 689, 32, 234, 936, 3642, 12987, 9131, 243, 251, 188, 1, 0, 0, 0, 0, 0, 2549, 23035, 1182, 363, 1069, 136, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 79, 2833, 25422, 0, 0, 0, 0, 0, 0)
y.norm = list(0.0, 0.0, 0.0, 0.018025939436226712, 0.0008371989288233016, 0.006122017167020393, 0.02448806866808157, 0.09528370308670202, 0.3397719527696318, 0.23888948184642397, 0.006357479365751946, 0.006566779097957772, 0.004918543706836896, 2.6162466525728176e-05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.06668812717408111, 0.6026524164201486, 0.030924035433410702, 0.009496975348839327, 0.02796767671600342, 0.0035580954474990316, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.002066834855532526, 0.07411826766738792, 0.6651022240170616, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
channels = c("Y", "Cb", "Cr")
plot.ycbcr = plot_feature_vector(y, y.norm, channels, image)
ggsave("featurevector_ycbcr.png", plot.ycbcr, width = PLOT_SIZE, height = PLOT_SIZE, units = "cm")
