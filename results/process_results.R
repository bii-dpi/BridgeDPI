library(glue)
library(dplyr)


get_row <- function(direction) {
  read.csv(glue("{direction}_testing_perf_123456789.csv")) %>%
    arrange(-AUPR) %>%
    mutate(model = "BridgeDPI", dir = direction) %>%
    select(dir, model, AUPR, recall_1:recall_50) %>%
    .[1,]
}

rbind(get_row("dtb"), get_row("btd")) %>%
  write.csv("../bridgedpi_results.csv", row.names = FALSE)
