# Bühlmann-Straub com actuar::cm, para conferir a implementação em Python.
#   Rscript r/buhlmann_straub.R entrada.csv saida.json
# entrada: colunas grupo, periodo, x (razão), w (peso)
suppressMessages({library(actuar); library(jsonlite)})
a <- commandArgs(trailingOnly = TRUE)
d <- read.csv(a[1], stringsAsFactors = FALSE)
periodos <- sort(unique(d$periodo))
grupos <- sort(unique(d$grupo))
X <- matrix(NA_real_, length(grupos), length(periodos))
W <- matrix(NA_real_, length(grupos), length(periodos))
for (r in seq_len(nrow(d))) {
  i <- match(d$grupo[r], grupos); j <- match(d$periodo[r], periodos)
  X[i, j] <- d$x[r]; W[i, j] <- d$w[r]
}
dados <- data.frame(grupo = grupos, X, W)
colnames(dados) <- c("grupo", paste0("x", seq_along(periodos)), paste0("w", seq_along(periodos)))
fit <- cm(~grupo, dados, ratios = 2:(1 + length(periodos)),
          weights = (2 + length(periodos)):(1 + 2 * length(periodos)), method = "Buhlmann-Gisler")
# fit$unbiased: no nível "portfolio" fica a variância ENTRE grupos (a); no nível do grupo, a variância
# DENTRO do grupo (s^2). Ex. (Hachemeister): portfolio = 89.639, state = 139.120.026.
u <- fit$unbiased
a_ <- unname(u["portfolio"])
s2 <- unname(u["grupo"])
writeLines(toJSON(list(s2 = s2, a = a_, cred = unname(fit$cred), media = unname(fit$means[[1]]),
                       medias_grupo = unname(fit$means[[2]]), grupos = grupos),
                  auto_unbox = TRUE, digits = NA), a[2])
