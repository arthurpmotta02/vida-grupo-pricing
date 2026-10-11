# Instala os pacotes R usados pelo projeto (rode uma vez, com R >= 4.2).
#   Rscript r/instalar_pacotes.R
pacotes <- c("glmmTMB", "lme4", "jsonlite", "actuar")
faltando <- setdiff(pacotes, rownames(installed.packages()))
if (length(faltando)) install.packages(faltando, repos = "https://cloud.r-project.org")
invisible(lapply(pacotes, library, character.only = TRUE))
cat("OK:", paste(sprintf("%s %s", pacotes, sapply(pacotes, function(p) as.character(packageVersion(p)))), collapse = " | "), "\n")
