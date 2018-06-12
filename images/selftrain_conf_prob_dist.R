require(ggplot2)
file2 = "st_pred_p_dist"
f2 = read.table(file2, header=FALSE, sep="\t", na.strings="NA", dec=".", strip.white=TRUE)
f2$V2=f2[order(f2$V2,decreasing = FALSE),]$V2

my2 = ggplot(f2, aes(f2$V1, f2$V2)) + geom_point() +
scale_x_continuous("Instance number") +
scale_y_continuous("Output probability") +
   theme_bw() +
   theme(plot.title = element_text(size=12),
	axis.text.x = element_text(size=12, angle=90, vjust=0.5),
	axis.text.y = element_text(size=12, angle=90, hjust=0.5),
	axis.title.x = element_text(size=12),
	axis.title.y = element_text(size=12, angle=90),
	panel.grid.minor = element_blank(),
	legend.justification=c(0,1), #left/right, bottom/top
	legend.position=c(0,1),
	legend.title = element_blank(),
	legend.text = element_text(size=12),
	legend.key.size = unit(1.4, "lines"),
	legend.key = element_rect(colour=NA),
	panel.border = element_blank())

ggsave("selftrain_conf_pred_dist_nonone.pdf", plot=my2, width = 7, height = 5)
