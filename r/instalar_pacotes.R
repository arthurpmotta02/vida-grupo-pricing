# Instala os pacotes R usados pelo projeto (rode uma vez, com R >= 4.2).
#   Rscript r/instalar_pacotes.R
# Instala na biblioteca pessoal do usuário: não exige permissão de administrador.
lib <- Sys.getenv("R_LIBS_USER")
dir.create(lib, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(lib, .libPaths()))
pacotes <- c("glmmTMB", "actuar", "jsonlite")
faltando <- setdiff(pacotes, rownames(installed.packages()))
if (length(faltando)) install.packages(faltando, lib = lib, repos = "https://cloud.r-project.org")
invisible(lapply(pacotes, library, character.only = TRUE))
cat("Biblioteca:", lib, "\n")
cat("OK:", paste(sprintf("%s %s", pacotes, sapply(pacotes, function(p) as.character(packageVersion(p)))), collapse = " | "), "\n")
