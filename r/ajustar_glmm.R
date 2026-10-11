# Ajuste de modelos de frequência de óbitos com glmmTMB (único pacote de modelagem do projeto).
# Chamado pelo Python (vgpricing.glmm):  Rscript r/ajustar_glmm.R especificacao.json
# A mesma função ajusta o GLMM da taxa manual e os modelos de comparação (Poisson e Binomial
# Negativa sem efeitos aleatórios), o que deixa as log-verossimilhanças na mesma base.
suppressMessages({
  library(glmmTMB)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
spec <- fromJSON(args[1], simplifyVector = TRUE)
dados <- read.csv(spec$entrada, stringsAsFactors = FALSE, colClasses = "character")
dados$N <- as.numeric(dados$N)
dados$E <- as.numeric(dados$E)
for (v in spec$numericas) dados[[v]] <- as.numeric(dados[[v]])
for (v in spec$categoricas) {
  dados[[v]] <- factor(dados[[v]])
  base <- spec$niveis_base[[v]]
  if (!is.null(base)) dados[[v]] <- relevel(dados[[v]], ref = base)
}
familia <- switch(spec$familia, poisson = poisson(), nbinom2 = nbinom2())

formula_de <- function(fixos, aleatorios) {
  termos <- c(if (length(fixos)) fixos else "1", aleatorios)
  as.formula(paste("N ~", paste(termos, collapse = " + "), "+ offset(log(E))"))
}
ajustar <- function(fixos, aleatorios) glmmTMB(formula_de(fixos, aleatorios), data = dados, family = familia)

saida_ajuste <- function(fit) {
  co <- summary(fit)$coefficients$cond
  vc <- VarCorr(fit)$cond
  re <- if (length(vc)) ranef(fit, condVar = TRUE)$cond else list()
  efeitos <- lapply(names(re), function(g) {
    m <- re[[g]]
    pv <- attr(m, "condVar")
    sd_cond <- if (is.list(pv)) sqrt(sapply(pv, function(a) a[1, 1])) else sqrt(pv[1, 1, ])
    list(nivel = rownames(m), modo = m[, 1], sd_condicional = as.numeric(sd_cond))
  })
  names(efeitos) <- names(re)
  # IC de 95% (Wald, escala log do desvio-padrão) de cada componente de variância
  ic <- if (length(vc)) tryCatch({
    ci <- confint(fit, parm = "theta_", method = "wald")
    list(termo = names(vc), var_inf = unname(ci[seq_along(vc), 1]^2), var_sup = unname(ci[seq_along(vc), 2]^2))
  }, error = function(e) NULL) else NULL
  list(
    formula = deparse(formula(fit), width.cutoff = 500L),
    familia = spec$familia,
    fixos = list(nome = rownames(co), estimativa = unname(co[, 1]), ep = unname(co[, 2])),
    vcov_fixos = unname(as.matrix(vcov(fit)$cond)),
    variancias = setNames(lapply(vc, function(x) unname(attr(x, "stddev")^2)), names(vc)),
    ic_variancias = ic,
    aleatorios = efeitos,
    loglik = as.numeric(logLik(fit)), aic = AIC(fit), gl = attr(logLik(fit), "df"),
    convergiu = (fit$fit$convergence == 0) && isTRUE(fit$sdr$pdHess)
  )
}

completo <- ajustar(spec$fixos, spec$aleatorios)
out <- list(modelo = saida_ajuste(completo), testes = list())
ll <- function(f) as.numeric(logLik(f))
gl <- function(f) attr(logLik(f), "df")
# razão de verossimilhança: retira um termo por vez do modelo completo
for (termo in spec$testar_fixos) {
  red <- ajustar(setdiff(spec$fixos, termo), spec$aleatorios)
  lr <- 2 * (ll(completo) - ll(red))
  out$testes[[length(out$testes) + 1]] <- list(termo = termo, tipo = "fixo", lr = lr, gl = gl(completo) - gl(red),
                                                p_valor = pchisq(max(lr, 0), gl(completo) - gl(red), lower.tail = FALSE))
}
for (termo in spec$testar_aleatorios) {
  red <- ajustar(spec$fixos, setdiff(spec$aleatorios, termo))
  lr <- 2 * (ll(completo) - ll(red))
  # variância na fronteira sob H0: mistura 50:50 de qui-quadrado(0) e qui-quadrado(1)
  out$testes[[length(out$testes) + 1]] <- list(termo = termo, tipo = "aleatorio", lr = lr, gl = 1,
                                                p_valor = 0.5 * pchisq(max(lr, 0), 1, lower.tail = FALSE))
}
writeLines(toJSON(out, auto_unbox = TRUE, digits = NA, null = "null"), spec$saida)
