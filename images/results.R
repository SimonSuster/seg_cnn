library(ggplot2)
file1 = "../result/results_test_trp"
f1 = read.table(file1, header=TRUE, sep=" ", na.strings="NA", dec=".", strip.white=TRUE)
#df$Names = factor(df$Names, levels=df[order(df$Proportion), "Names"])
f1$model = factor(f1$model, levels=f1[order(f1$F1), "model"])
my= ggplot() +
geom_errorbar(data=f1, mapping=aes(x=model, ymin=upper, ymax=lower), width=0.2, size=0.5, color="black") +
geom_point(data=f1, mapping=aes(x=model, y=F1), size=4, shape=21, fill="white") +
scale_y_continuous(limits = c(60,90)) +
annotate("text", x = f1$model, y=f1$F1, label = f1$F1, hjust = 1.2, colour = "black", size=7.5) +
   theme_bw() +
   theme(plot.title = element_text(size=20),
	axis.text.x = element_text(color="black", size=20, vjust=0.5),
	axis.text.y = element_text(color="black", size=20, angle=90, hjust=0.5),
	axis.title.x = element_text(color="black", size=20),
	axis.title.y = element_text(color="black", size=20, angle=90),
	panel.grid.minor = element_blank(),
	legend.justification=c(0,1), #left/right, bottom/top
	legend.position=c(0,1),
	legend.title = element_blank(),
	legend.text = element_text(size=12),
	legend.key.size = unit(1.4, "lines"),
	legend.key = element_rect(colour=NA),
	panel.border = element_blank())
ggsave("results_test_trp.pdf", plot=my, width = 7, height = 5)

file1 = "../result/results_test_tep"
f1 = read.table(file1, header=TRUE, sep=" ", na.strings="NA", dec=".", strip.white=TRUE)
#df$Names = factor(df$Names, levels=df[order(df$Proportion), "Names"])
#f1$model = factor(f1$model, levels=f1[order(f1$F1), "model"])
f1$model = factor(f1$model, levels=f1[c(1,2,3), "model"])
my= ggplot() +
geom_errorbar(data=f1, mapping=aes(x=model, ymin=upper, ymax=lower), width=0.2, size=0.5, color="black") +
geom_point(data=f1, mapping=aes(x=model, y=F1), size=4, shape=21, fill="white") +
scale_y_continuous(limits = c(60,90)) +
annotate("text", x = f1$model, y=f1$F1, label = f1$F1, hjust = 1.3, colour = "black", size=7.5) +
   theme_bw() +
   theme(plot.title = element_text(size=20),
	axis.text.x = element_text(color="black", size=20, vjust=0.5),
	axis.text.y = element_text(color="black", size=20, angle=90, hjust=0.5),
	axis.title.x = element_text(color="black", size=20),
	axis.title.y = element_text(color="black", size=20, angle=90),
	panel.grid.minor = element_blank(),
	legend.justification=c(0,1), #left/right, bottom/top
	legend.position=c(0,1),
	legend.title = element_blank(),
	legend.text = element_text(size=12),
	legend.key.size = unit(1.4, "lines"),
	legend.key = element_rect(colour=NA),
	panel.border = element_blank())
ggsave("results_test_tep.pdf", plot=my, width = 7, height = 5)

file1 = "../result/results_test_pp"
f1 = read.table(file1, header=TRUE, sep=" ", na.strings="NA", dec=".", strip.white=TRUE)
#df$Names = factor(df$Names, levels=df[order(df$Proportion), "Names"])
f1$model = factor(f1$model, levels=f1[order(f1$F1), "model"])
my= ggplot() +
geom_errorbar(data=f1, mapping=aes(x=model, ymin=upper, ymax=lower), width=0.2, size=0.5, color="black") +
geom_point(data=f1, mapping=aes(x=model, y=F1), size=4, shape=21, fill="white") +
scale_y_continuous(limits = c(60,90)) +
annotate("text", x = f1$model, y=f1$F1, label = f1$F1, hjust = 1.2, colour = "black", size=7.5) +
   theme_bw() +
   theme(plot.title = element_text(size=20),
	axis.text.x = element_text(color="black", size=20, vjust=0.5),
	axis.text.y = element_text(color="black", size=20, angle=90, hjust=0.5),
	axis.title.x = element_text(color="black", size=20),
	axis.title.y = element_text(color="black", size=20, angle=90),
	panel.grid.minor = element_blank(),
	legend.justification=c(0,1), #left/right, bottom/top
	legend.position=c(0,1),
	legend.title = element_blank(),
	legend.text = element_text(size=12),
	legend.key.size = unit(1.4, "lines"),
	legend.key = element_rect(colour=NA),
	panel.border = element_blank())
ggsave("results_test_pp.pdf", plot=my, width = 7, height = 5)