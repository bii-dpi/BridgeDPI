library(glue)
library(tidyr)
library(dplyr)
library(ggplot2)


NUM_EXPTS <- 9
COLORS <- c("19" = "#faef60", "49" = "orange", "94" = "#e74645")


read_pathway <- function(expt_number, statistic, num_testing_pdbs) {
  data <- read.csv("../results/validation_results.csv")
  data <- data[c("direction", statistic)]
  data %>%
    filter(direction %in% glue("pathway{num_testing_pdbs}_{seq(15, 135, 15)}_{expt_number}")) %>%
    mutate(expt_number = expt_number,
           num_training_pdbs = lapply(direction, function(dir) { strsplit(dir, "_")[[1]][2] })) %>%
    select(-direction)
}


get_mean_pathway_data <- function(statistic, num_testing_pdbs) {
  all_pathways <- 
    sapply(0:NUM_EXPTS, function(expt_num) { 
      glue("pathway{num_testing_pdbs}_{seq(15, 135, 15)}_{expt_num}")
    }) %>%
    c
  raw_data <- read.csv("../results/validation_results.csv") %>%
    filter(direction %in% all_pathways) %>%
    mutate(num_training_pdbs = sapply(strsplit(direction, "_"),
                                      function(pathway) as.numeric(pathway[2]))) %>%
    select(num_training_pdbs, !!sym(statistic)) %>%
    group_by(num_training_pdbs) %>%
    summarize(mean_ = mean(!!sym(statistic)),
              sd_ = sd(!!sym(statistic)),
              min_ = mean_ - sd_,
              max_ = mean_ + sd_)
}


plot_statistic <- function(statistic) {
  if (startsWith(statistic, "recall")) {
    title <- paste0(paste(strsplit(statistic, "_")[[1]], collapse=" (precision "), "%)")
  } else {
    title <- statistic
  }
  
  mean_data <- NULL
  for (num_testing_pdbs in c(19, 49, 94)) {
    mean_data <- rbind(mean_data,
                       get_mean_pathway_data(statistic, num_testing_pdbs) %>%
                         mutate(num_testing_pdbs = as.character(num_testing_pdbs)))
  }
  
  names(mean_data) <- c("num_training_pdbs",
                        glue("{statistic}_{c('mean', 'sd', 'min', 'max')}"),
                        "num_testing_pdbs")
  
  plot <- ggplot() +
    geom_smooth(data = mean_data,
                aes_string(x = "num_training_pdbs", y = glue("{statistic}_mean"),
                           color = "num_testing_pdbs"),
                se = F, size = 2) +
    theme_classic() +
    ylab(glue("External validation {title}")) +
    xlab("Number of training proteins") +
    ggtitle(glue("BridgeDPI {title}")) +
    theme(legend.position = "bottom") +
    scale_x_continuous(breaks = seq(15, 135, 15)) +
    scale_color_manual(values = COLORS)
  
  ggsave(glue("results/pathway_{statistic}_bridgedpi_mean.png"),
         plot, device = "png", height=8.29/2.5, width=9.5/1.5)
  
  print(plot)
}

for (statistic in c("AUPR", glue("recall_{c(1, 5, 10, 25, 50)}"))) {
  plot_statistic(statistic)
}  
