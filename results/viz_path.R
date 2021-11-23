library(dplyr)
library(glue)
library(ggplot2)


DIRECTIONS <- c("Training: DUD-E, external val.: BindingDB",
                "Training: BindingDB, external val.: DUD-E")
names(DIRECTIONS) <- c("dztbz", "bztdz")

SEEDS <- c("0", "1", "2")
names(SEEDS) <- c("123456789",
                  "878912589",
                  "537912576")

MODES <- c("Training", "External validation")
names(MODES) <- c("training", "testing")

METRICS <- c("Accuracy", "AUC", "Precision", "Recall", "AUPR", "F1", "BCE")


read_mode_seed <- function(direction, mode_, seed) {
  read.csv(glue("{direction}_{mode_}_perf_{seed}.csv"),
           header = FALSE,
           skip = 1,
           col.names = METRICS) %>%
    mutate(Epoch = 1:n(),
           mode_ = MODES[mode_],
           Seed = SEEDS[seed])
}


read_direction <- function(direction) {
  data <- NULL
  for (seed in names(SEEDS)) {
    for (mode_ in names(MODES)) {
      data <- rbind(data, read_mode_seed(direction, mode_, seed))
    }
  }
  mutate(data, mode_ = factor(mode_, levels = MODES))
}


plot_direction_stat <- function(direction, data, metric) {
  peak_fn <- ifelse(metric == "BCE", min, max)
  
  peak <- data %>%
    filter(mode_ == "External validation") %>%
    .[[metric]] %>%
    peak_fn(.) %>%
    round(3)
  
  plot <- ggplot(data, aes_string(x = "Epoch", y = metric, color = "Seed")) +
    geom_line(size = 1.2) +
    theme_classic() +
    ggtitle(glue("{metric} (best val.: {peak})\n{DIRECTIONS[direction]}")) +
    facet_grid(mode_ ~ .) +
    ylab(glue("{metric}")) +
    theme(legend.position = "bottom")
  
  ggsave(glue("viz/{direction}_{metric}.pdf"),
         plot, device = "pdf", height=8.29/2, width=9.5/1.3)
  ggsave(glue("viz/{direction}_{metric}.png"),
         plot, device = "pdf", height=8.29/2, width=9.5/1.3)
  
  print(plot)
}


plot_direction <- function(direction) {
  data <- read_direction(direction)
  for (metric in METRICS) {
    plot_direction_stat(direction, data, metric)
  }
}


plot_direction("dztbz")
plot_direction("bztdz")
