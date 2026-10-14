# GAM de Poisson com offset (mgcv), suavidade escolhida por REML.
#   Rscript r/ajustar_gam.R especificacao.json
suppressMessages({library(mgcv); library(jsonlite)})
spec <- fromJSON(commandArgs(trailingOnly = TRUE)[1], simplifyVector = TRUE)
ler <- function(p) read.csv(p, stringsAsFactors = FALSE, colClasses = "character")
tr <- ler(spec$treino)
prever <- lapply(spec$prever, ler)
prep <- function(d, niveis = NULL) {
  for (v in spec$numericas) d[[v]] <- as.numeric(d[[v]])
  for (v in spec$categoricas) d[[v]] <- factor(d[[v]], levels = if (is.null(niveis)) sort(unique(d[[v]])) else niveis[[v]])
  d$N <- as.numeric(d$N); d$E <- as.numeric(d$E); d
}
tr <- prep(tr)
niveis <- lapply(spec$categoricas, function(v) levels(tr[[v]])); names(niveis) <- spec$categoricas
suaves <- sprintf("s(%s, k = %d)", spec$suaves, as.integer(spec$k))
f <- as.formula(paste("N ~", paste(c(suaves, spec$categoricas), collapse = " + "), "+ offset(log(E))"))
# bam: a versão do mgcv para bases grandes (mesmo modelo; suavidade por fREML)
fit <- bam(f, data = tr, family = poisson, method = "fREML", discrete = TRUE)
out <- list(formula = deparse(f, width.cutoff = 500L), edf = unname(summary(fit)$s.table[, "edf"]),
            p_suave = unname(summary(fit)$s.table[, "p-value"]), deviance = deviance(fit), aic = AIC(fit),
            convergiu = isTRUE(fit$converged), previsoes = list(), curvas = list())
for (nome in names(prever)) {
  d <- prep(prever[[nome]], niveis)
  arq <- file.path(spec$pasta_saida, paste0("prev_", nome, ".csv"))
  write.csv(data.frame(mu = as.numeric(predict(fit, newdata = d, type = "response"))), arq, row.names = FALSE)
  out$previsoes[[nome]] <- arq
}
# curva de cada termo suave (escala log, centrada) com erro-padrão, numa grade do treino
for (s in spec$suaves) {
  g <- tr[rep(1, 100), ]
  g[[s]] <- seq(min(tr[[s]]), max(tr[[s]]), length.out = 100)
  p <- predict(fit, newdata = g, type = "terms", se.fit = TRUE)
  col <- grep(paste0("s(", s, ")"), colnames(p$fit), fixed = TRUE)
  out$curvas[[s]] <- list(x = g[[s]], efeito = unname(p$fit[, col]), ep = unname(p$se.fit[, col]))
}
writeLines(toJSON(out, auto_unbox = TRUE, digits = NA), spec$saida)
