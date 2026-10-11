# Bühlmann-Straub com actuar::cm (estimadores de Bühlmann-Gisler).
#   Rscript r/buhlmann_straub.R entrada.csv saida.json
# entrada: colunas grupo, periodo, x (razão observada), w (peso)
suppressMessages({library(actuar); library(jsonlite)})
a <- commandArgs(trailingOnly = TRUE)
d <- read.csv(a[1], stringsAsFactors = FALSE, colClasses = c(grupo = "character"))
periodos <- sort(unique(d$periodo))
grupos <- sort(unique(d$grupo))
X <- matrix(NA_real_, length(grupos), length(periodos))
W <- matrix(NA_real_, length(grupos), length(periodos))
X[cbind(match(d$grupo, grupos), match(d$periodo, periodos))] <- d$x
W[cbind(match(d$grupo, grupos), match(d$periodo, periodos))] <- d$w
dados <- data.frame(grupo = grupos, X, W)
n <- length(periodos)
colnames(dados) <- c("grupo", paste0("x", seq_len(n)), paste0("w", seq_len(n)))
fit <- cm(~grupo, dados, ratios = 2:(1 + n), weights = (2 + n):(1 + 2 * n), method = "Buhlmann-Gisler")
# fit$unbiased: nível "portfolio" = variância ENTRE grupos (a); nível do grupo = variância DENTRO (s^2)
u <- fit$unbiased
writeLines(toJSON(list(a = unname(u["portfolio"]), s2 = unname(u["grupo"]), grupos = grupos,
                       Z = unname(fit$cred), media_grupo = unname(fit$means[[2]]),
                       media_carteira = unname(fit$means[[1]])),
                  auto_unbox = TRUE, digits = NA), a[2])
