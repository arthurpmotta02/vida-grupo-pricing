# Ajuste do GLMM de frequência de óbitos (taxa manual) com glmmTMB.
# Chamado pelo Python (vgpricing.glmm) com um arquivo JSON de especificação:
#   Rscript r/ajustar_glmm.R especificacao.json
# Entrada: CSV de células com N (óbitos), E (esperados pela tábua) e covariáveis.
# Saída: JSON com efeitos fixos, componentes de variância, efeitos aleatórios (modos e
# desvios condicionais), log-verossimilhança, AIC, testes de razão de verossimilhança e,
# opcionalmente, a conferência com lme4::glmer.
suppressMessages({
  library(glmmTMB)
  library(lme4)
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

formula_de <- function(fixos, aleatorios) {
  termos <- c(if (length(fixos)) fixos else "1", aleatorios)
  as.formula(paste("N ~", paste(termos, collapse = " + "), "+ offset(log(E))"))
}

ajustar <- function(fixos, aleatorios) {
  glmmTMB(formula_de(fixos, aleatorios), data = dados, family = poisson)
}

saida_ajuste <- function(fit) {
  co <- summary(fit)$coefficients$cond
  vc <- VarCorr(fit)$cond
  re <- ranef(fit, condVar = TRUE)$cond
  efeitos <- lapply(names(re), function(g) {
    m <- re[[g]]
    pv <- attr(m, "condVar")
    sd_cond <- if (is.list(pv)) sqrt(sapply(pv, function(a) a[1, 1])) else sqrt(pv[1, 1, ])
    list(grupo = g, nivel = rownames(m), modo = m[, 1], sd_condicional = as.numeric(sd_cond))
  })
  names(efeitos) <- names(re)
  # IC de 95% (Wald, escala log do desvio-padrão) para cada componente de variância
  ic <- tryCatch({
    ci <- confint(fit, parm = "theta_", method = "wald")
    list(termo = sub("^Std\\.Dev\\.\\(Intercept\\)\\|", "", rownames(ci)),
         var_inf = unname(ci[, 1]^2), var_sup = unname(ci[, 2]^2))
  }, error = function(e) NULL)
  list(
    formula = deparse(formula(fit), width.cutoff = 500L),
    ic_variancias = ic,
    fixos = list(nome = rownames(co), estimativa = unname(co[, 1]), ep = unname(co[, 2])),
    vcov_fixos = unname(as.matrix(vcov(fit)$cond)),
    variancias = setNames(lapply(vc, function(x) unname(attr(x, "stddev")^2)), names(vc)),
    aleatorios = efeitos,
    loglik = as.numeric(logLik(fit)), aic = AIC(fit), gl = attr(logLik(fit), "df"),
    n = nobs(fit),
    convergiu = (fit$fit$convergence == 0) && isTRUE(fit$sdr$pdHess)
  )
}

fixos <- spec$fixos
aleatorios <- spec$aleatorios
completo <- ajustar(fixos, aleatorios)
out <- list(modelo = saida_ajuste(completo), testes = list())

# razão de verossimilhança: retira um termo por vez do modelo completo
for (termo in spec$testar_fixos) {
  red <- ajustar(setdiff(fixos, termo), aleatorios)
  lr <- 2 * (as.numeric(logLik(completo)) - as.numeric(logLik(red)))
  gl <- attr(logLik(completo), "df") - attr(logLik(red), "df")
  out$testes[[length(out$testes) + 1]] <- list(termo = termo, tipo = "fixo", lr = lr, gl = gl,
                                                p_valor = pchisq(max(lr, 0), gl, lower.tail = FALSE))
}
for (termo in spec$testar_aleatorios) {
  red <- ajustar(fixos, setdiff(aleatorios, termo))
  lr <- 2 * (as.numeric(logLik(completo)) - as.numeric(logLik(red)))
  # variância na fronteira sob H0: mistura 50:50 de qui-quadrado(0) e qui-quadrado(1)
  out$testes[[length(out$testes) + 1]] <- list(termo = termo, tipo = "aleatorio", lr = lr, gl = 1,
                                                p_valor = 0.5 * pchisq(max(lr, 0), 1, lower.tail = FALSE))
}

if (isTRUE(spec$conferir_lme4)) {
  g <- glmer(formula_de(fixos, aleatorios), data = dados, family = poisson,
             control = glmerControl(optimizer = "bobyqa"))
  vc <- VarCorr(g)
  out$lme4 <- list(fixos = list(nome = names(fixef(g)), estimativa = unname(fixef(g))),
                   variancias = setNames(lapply(vc, function(x) unname(attr(x, "stddev")^2)), names(vc)),
                   loglik = as.numeric(logLik(g)))
}

writeLines(toJSON(out, auto_unbox = TRUE, digits = NA, null = "null"), spec$saida)
