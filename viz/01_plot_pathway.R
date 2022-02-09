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


get_indiv_pathway_data <- function(statistic, num_testing_pdbs) {
  results <- NULL
  for (expt_number in 0:NUM_EXPTS) {
    results <- rbind(results, read_pathway(expt_number, statistic, num_testing_pdbs))
  }
  mutate(results,
         expt_number = factor(expt_number, levels = 0:NUM_EXPTS),
         num_training_pdbs = as.numeric(num_training_pdbs))
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


plot_statistic <- function(statistic, num_testing_pdbs) {
  if (startsWith(statistic, "recall")) {
    title <- paste0(paste(strsplit(statistic, "_")[[1]], collapse=" (precision "), "%)")
  } else {
    title <- statistic
  }
  
  indiv_data <- get_indiv_pathway_data(statistic, num_testing_pdbs)
  mean_data <- get_mean_pathway_data(statistic, num_testing_pdbs)
  names(mean_data) <- c("num_training_pdbs", glue("{statistic}_{c('mean', 'sd', 'min', 'max')}"))
  
  plot <- ggplot() +
    geom_smooth(data = indiv_data,
                aes_string(x = "num_training_pdbs", y = statistic,
                           group = "expt_number"),
                se = F, color = "lightgray") +
    geom_smooth(data = mean_data,
                aes_string(x = "num_training_pdbs", y = glue("{statistic}_mean")),
                se = F, color = COLORS[as.character(num_testing_pdbs)], size = 2.5) +
    geom_ribbon(data = mean_data,
                aes_string(x = "num_training_pdbs",
                           ymin = glue("{statistic}_min"),
                           ymax = glue("{statistic}_max")),
                alpha = 0.2,
                fill = "blue") +
    theme_classic() +
    ylab(glue("External validation {title}")) +
    xlab("Number of training proteins") +
    ggtitle(glue("BridgeDPI, {num_testing_pdbs} testing proteins")) +
    theme(legend.position = "none") +
    scale_x_continuous(breaks = seq(15, 135, 15))
  
  ggsave(glue("results/pathway_{statistic}_{num_testing_pdbs}_bridgedpi.png"),
         plot, device = "png", height=8.29/2.5, width=9.5/1.5)
  
  print(plot)
}

for (num_testing_pdbs in c(19, 49, 94)) {
  for (statistic in c("AUPR", glue("recall_{c(1, 5, 10, 25, 50)}"))) {
    plot_statistic(statistic, num_testing_pdbs)
  }  
}
