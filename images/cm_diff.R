library(ggplot2)
library(stringr)  # for having \n in legend title

# run data_util.py first to obtain the *_cm files

file1 = "../result/compa_trp_cm"
file2 = "../result/segcnn_trp_cm"
f1 = read.table(file1, header=TRUE, sep="\t", na.strings="NA", dec=".", strip.white=TRUE)
f2 = read.table(file2, header=TRUE, sep="\t", na.strings="NA", dec=".", strip.white=TRUE)
f_diff = data.frame(Gold=f1$Gold, System=f1$System, value=round(f1$value - f2$value, 1))
cm = ggplot(data = f_diff, aes(x=System, y=Gold, fill=value)) +
  geom_tile(color="white") +
  scale_fill_gradient2(low = "green", high = "red", mid = "white", space = "Lab", name="Comp\n - SegCNN") +
  theme_minimal()+
  theme(plot.title = element_text(color="black", size=20),
	axis.text.x = element_text(color="black", size=18, vjust=0.5),
	axis.text.y = element_text(color="black", size=18, angle=90, hjust=0.5),
	axis.title.x = element_text(color="black", size=20),
	axis.title.y = element_text(color="black", size=20, angle=90),
	legend.text = element_text(color="black", size=16),
    legend.title= element_text(color="black", size=16)) +
  coord_fixed() +
  geom_text(aes(System, Gold, label = value), color = "black", size = 6) +
  scale_y_discrete(limits = rev(levels(f_diff$Gold))) +
  scale_x_discrete(position = "top")
ggsave("compa_trp_cm_diff.pdf", plot=cm, width = 7, height = 5)

file1 = "../result/compa_tep_cm"
file2 = "../result/segcnn_tep_cm"
f1 = read.table(file1, header=TRUE, sep="\t", na.strings="NA", dec=".", strip.white=TRUE)
f2 = read.table(file2, header=TRUE, sep="\t", na.strings="NA", dec=".", strip.white=TRUE)
f_diff = data.frame(Gold=f1$Gold, System=f1$System, value=round(f1$value - f2$value, 1))
cm = ggplot(data = f_diff, aes(x=System, y=Gold, fill=value)) +
  geom_tile(color="white") +
  scale_fill_gradient2(low = "green", high = "red", mid = "white", space = "Lab", name="Comp\n - SegCNN") +
  theme_minimal()+
    theme(plot.title = element_text(color="black", size=20),
	axis.text.x = element_text(color="black", size=18, vjust=0.5),
	axis.text.y = element_text(color="black", size=18, angle=90, hjust=0.5),
	axis.title.x = element_text(color="black", size=20),
	axis.title.y = element_text(color="black", size=20, angle=90),
	legend.text = element_text(color="black", size=16),
    legend.title= element_text(color="black", size=16)) +
  coord_fixed() +
  geom_text(aes(System, Gold, label = value), color = "black", size = 7) +
  scale_y_discrete(limits = rev(levels(f_diff$Gold))) +
  scale_x_discrete(position = "top")
ggsave("compa_tep_cm_diff.pdf", plot=cm, width = 7, height = 5)

file1 = "../result/compa_pp_cm"
file2 = "../result/segcnn_pp_cm"
f1 = read.table(file1, header=TRUE, sep="\t", na.strings="NA", dec=".", strip.white=TRUE)
f2 = read.table(file2, header=TRUE, sep="\t", na.strings="NA", dec=".", strip.white=TRUE)
f_diff = data.frame(Gold=f1$Gold, System=f1$System, value=round(f1$value - f2$value, 1))
cm = ggplot(data = f_diff, aes(x=System, y=Gold, fill=value)) +
  geom_tile(color="white") +
  scale_fill_gradient2(low = "green", high = "red", mid = "white", space = "Lab", name="Comp\n - SegCNN") +
  theme_minimal()+
    theme(plot.title = element_text(color="black", size=20),
	axis.text.x = element_text(color="black", size=18, vjust=0.5),
	axis.text.y = element_text(color="black", size=18, angle=90, hjust=0.5),
	axis.title.x = element_text(color="black", size=20),
	axis.title.y = element_text(color="black", size=20, angle=90),
	legend.text = element_text(color="black", size=16),
    legend.title= element_text(color="black", size=16)) +
  coord_fixed() +
  geom_text(aes(System, Gold, label = value), color = "black", size = 8) +
  scale_y_discrete(limits = rev(levels(f_diff$Gold))) +
  scale_x_discrete(position = "top")
ggsave("compa_pp_cm_diff.pdf", plot=cm, width = 7, height = 5)



file1 = "../result/semclass_trp_cm"
file2 = "../result/segcnn_trp_cm"
f1 = read.table(file1, header=TRUE, sep="\t", na.strings="NA", dec=".", strip.white=TRUE)
f2 = read.table(file2, header=TRUE, sep="\t", na.strings="NA", dec=".", strip.white=TRUE)
f_diff = data.frame(Gold=f1$Gold, System=f1$System, value=round(f1$value - f2$value, 1))
cm = ggplot(data = f_diff, aes(x=System, y=Gold, fill=value)) +
  geom_tile(color="white") +
  scale_fill_gradient2(low = "green", high = "red", mid = "white", space = "Lab", name="SemClass\n - SegCNN") +
  theme_minimal()+
    theme(plot.title = element_text(color="black", size=20),
	axis.text.x = element_text(color="black", size=18, vjust=0.5),
	axis.text.y = element_text(color="black", size=18, angle=90, hjust=0.5),
	axis.title.x = element_text(color="black", size=20),
	axis.title.y = element_text(color="black", size=20, angle=90),
	legend.text = element_text(color="black", size=16),
    legend.title= element_text(color="black", size=16)) +
  coord_fixed() +
  geom_text(aes(System, Gold, label = value), color = "black", size = 6) +
  scale_y_discrete(limits = rev(levels(f_diff$Gold))) +
  scale_x_discrete(position = "top")
ggsave("semclass_trp_cm_diff.pdf", plot=cm, width = 7, height = 5)

file1 = "../result/semclass_tep_cm"
file2 = "../result/segcnn_tep_cm"
f1 = read.table(file1, header=TRUE, sep="\t", na.strings="NA", dec=".", strip.white=TRUE)
f2 = read.table(file2, header=TRUE, sep="\t", na.strings="NA", dec=".", strip.white=TRUE)
f_diff = data.frame(Gold=f1$Gold, System=f1$System, value=round(f1$value - f2$value, 1))
cm = ggplot(data = f_diff, aes(x=System, y=Gold, fill=value)) +
  geom_tile(color="white") +
  scale_fill_gradient2(low = "green", high = "red", mid = "white", space = "Lab", name="SemClass\n - SegCNN") +
  theme_minimal()+
    theme(plot.title = element_text(color="black", size=20),
	axis.text.x = element_text(color="black", size=18, vjust=0.5),
	axis.text.y = element_text(color="black", size=18, angle=90, hjust=0.5),
	axis.title.x = element_text(color="black", size=20),
	axis.title.y = element_text(color="black", size=20, angle=90),
	legend.text = element_text(color="black", size=16),
    legend.title= element_text(color="black", size=16)) +
  coord_fixed() +
  geom_text(aes(System, Gold, label = value), color = "black", size = 7) +
  scale_y_discrete(limits = rev(levels(f_diff$Gold))) +
  scale_x_discrete(position = "top")
ggsave("semclass_tep_cm_diff.pdf", plot=cm, width = 7, height = 5)


file1 = "../result/pmi_trp_cm"
file2 = "../result/segcnn_trp_cm"
f1 = read.table(file1, header=TRUE, sep="\t", na.strings="NA", dec=".", strip.white=TRUE)
f2 = read.table(file2, header=TRUE, sep="\t", na.strings="NA", dec=".", strip.white=TRUE)
f_diff = data.frame(Gold=f1$Gold, System=f1$System, value=round(f1$value - f2$value, 1))
cm = ggplot(data = f_diff, aes(x=System, y=Gold, fill=value)) +
  geom_tile(color="white") +
  scale_fill_gradient2(low = "green", high = "red", mid = "white", space = "Lab", name="PMI\n - SegCNN") +
  theme_minimal()+
    theme(plot.title = element_text(color="black", size=20),
	axis.text.x = element_text(color="black", size=18, vjust=0.5),
	axis.text.y = element_text(color="black", size=18, angle=90, hjust=0.5),
	axis.title.x = element_text(color="black", size=20),
	axis.title.y = element_text(color="black", size=20, angle=90),
	legend.text = element_text(color="black", size=16),
    legend.title= element_text(color="black", size=16)) +
  coord_fixed() +
  geom_text(aes(System, Gold, label = value), color = "black", size = 6) +
  scale_y_discrete(limits = rev(levels(f_diff$Gold))) +
  scale_x_discrete(position = "top")
ggsave("pmi_trp_cm_diff.pdf", plot=cm, width = 7, height = 5)
file1 = "../result/pmi_tep_cm"
file2 = "../result/segcnn_tep_cm"
f1 = read.table(file1, header=TRUE, sep="\t", na.strings="NA", dec=".", strip.white=TRUE)
f2 = read.table(file2, header=TRUE, sep="\t", na.strings="NA", dec=".", strip.white=TRUE)
f_diff = data.frame(Gold=f1$Gold, System=f1$System, value=round(f1$value - f2$value, 1))
cm = ggplot(data = f_diff, aes(x=System, y=Gold, fill=value)) +
  geom_tile(color="white") +
  scale_fill_gradient2(low = "green", high = "red", mid = "white", space = "Lab", name="PMI\n - SegCNN") +
  theme_minimal()+
    theme(plot.title = element_text(color="black", size=20),
	axis.text.x = element_text(color="black", size=18, vjust=0.5),
	axis.text.y = element_text(color="black", size=18, angle=90, hjust=0.5),
	axis.title.x = element_text(color="black", size=20),
	axis.title.y = element_text(color="black", size=20, angle=90),
	legend.text = element_text(color="black", size=16),
    legend.title= element_text(color="black", size=16)) +
  coord_fixed() +
  geom_text(aes(System, Gold, label = value), color = "black", size = 7) +
  scale_y_discrete(limits = rev(levels(f_diff$Gold))) +
  scale_x_discrete(position = "top")
ggsave("pmi_tep_cm_diff.pdf", plot=cm, width = 7, height = 5)
file1 = "../result/pmi_pp_cm"
file2 = "../result/segcnn_pp_cm"
f1 = read.table(file1, header=TRUE, sep="\t", na.strings="NA", dec=".", strip.white=TRUE)
f2 = read.table(file2, header=TRUE, sep="\t", na.strings="NA", dec=".", strip.white=TRUE)
f_diff = data.frame(Gold=f1$Gold, System=f1$System, value=round(f1$value - f2$value, 1))
cm = ggplot(data = f_diff, aes(x=System, y=Gold, fill=value)) +
  geom_tile(color="white") +
  scale_fill_gradient2(low = "green", high = "red", mid = "white", space = "Lab", name="PMI\n - SegCNN") +
  theme_minimal()+
    theme(plot.title = element_text(color="black", size=20),
	axis.text.x = element_text(color="black", size=18, vjust=0.5),
	axis.text.y = element_text(color="black", size=18, angle=90, hjust=0.5),
	axis.title.x = element_text(color="black", size=20),
	axis.title.y = element_text(color="black", size=20, angle=90),
	legend.text = element_text(color="black", size=16),
    legend.title= element_text(color="black", size=16)) +
  coord_fixed() +
  geom_text(aes(System, Gold, label = value), color = "black", size = 8) +
  scale_y_discrete(limits = rev(levels(f_diff$Gold))) +
  scale_x_discrete(position = "top")

ggsave("pmi_pp_cm_diff.pdf", plot=cm, width = 7, height = 5)



