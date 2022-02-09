library(glue)
library(tidyr)
library(dplyr)
library(ggplot2)

MODE <- "protein"
NUM_EXPTS <- 9

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


get_indiv_sims <- function() {
  read.csv(glue("../../get_data/Shallow/sim_expts/pathway_{MODE}_sims.csv")) %>%
    mutate(expt_number = factor(expt_number, levels = 0:NUM_EXPTS),
           num_training_pdbs = as.numeric(num_training_pdbs))
}


plot_statistic <- function(statistic, num_testing_pdbs) {
  if (startsWith(statistic, "recall")) {
    title <- paste0(paste(strsplit(statistic, "_")[[1]], collapse=" (precision "), "%)")
  } else {
    title <- statistic
  }
  
  perf_data <- get_indiv_pathway_data(statistic, num_testing_pdbs)
  sim_data <- get_indiv_sims()
  merged_df <- left_join(perf_data, sim_data)
  
  correl <- cor(merged_df["mean_sim"], merged_df[statistic])
  correl <- round(correl, 3)
  
  if (MODE == "actives") {
    tag <- "actives similarity (Tc)"
  } else {
    tag <- "protein identicality (%)"
  }
  
  plot <- ggplot(data = merged_df,
                 aes_string(x = "mean_sim", y = glue("{statistic}"))) +
    geom_point() +
    geom_smooth(color="red",method = "lm", se=F) +
    theme_classic() +
    ylab(glue("External validation {title}")) +
    xlab(glue("Mean {tag} between training and external validation")) +
    theme(legend.position = "none") +
    ggtitle(glue("{title} vs. mean {tag} (R-square: {correl})"))
  
  ggsave(glue("results/{MODE}_vs_{statistic}_bridgedpi.png"),
         plot, device = "png", height=8.29/2, width=9.5/1.25)
  
  print(plot)
}


for (num_testing_pdbs in c(19, 49, 94)) {
  for (statistic in c("AUPR", glue("recall_{c(1, 5, 10, 25, 50)}"))) {
    plot_statistic(statistic, num_testing_pdbs)
  }  
}

