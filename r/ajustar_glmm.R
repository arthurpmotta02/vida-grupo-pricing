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
TOL_VARIANCIA <- 1e-8   # abaixo disso a variância é zero numérico (ajuste singular)

variancias_de <- function(fit) {
  vc <- tryCatch(VarCorr(fit)$cond, error = function(e) NULL)
  if (is.null(vc) || !length(vc)) return(numeric(0))
  sapply(vc, function(x) unname(attr(x, "stddev")^2))
}

# Limites de iteração maiores que o padrão do nlminb: com dezenas de parâmetros e centenas de milhares de linhas,
# o padrão para antes de convergir (código 9). Se ainda assim não convergir, tenta o otimizador BFGS
# (as duas medidas são as recomendadas no guia de solução de problemas do glmmTMB).
CONTROLE <- glmmTMBControl(optCtrl = list(iter.max = 1e4, eval.max = 1e4))
ajustar_bruto <- function(fixos, aleatorios) {
  f <- formula_de(fixos, aleatorios)
  fit <- suppressWarnings(glmmTMB(f, data = dados, family = familia, control = CONTROLE))
  if (fit$fit$convergence != 0) {
    fit2 <- suppressWarnings(glmmTMB(f, data = dados, family = familia,
                                     control = glmmTMBControl(optimizer = optim, optArgs = list(method = "BFGS"))))
    if (fit2$fit$convergence == 0) fit <- fit2
  }
  fit
}

# Ajusta e, se o ajuste for singular (variância nula de algum efeito aleatório) ou a log-verossimilhança
# não for finita, retira os efeitos com variância nula e reajusta. Versões diferentes do glmmTMB tratam o
# ajuste singular de formas diferentes (umas devolvem variância ~0, outras NA); esta regra dá o mesmo
# resultado em todas. Devolve o ajuste e os termos retirados.
ajustar <- function(fixos, aleatorios) {
  removidos <- character(0)
  repeat {
    fit <- ajustar_bruto(fixos, aleatorios)
    if (!length(aleatorios)) break
    v <- variancias_de(fit)
    nomes_termo <- sub("^\\(1 *\\| *(.*)\\)$", "\\1", gsub(" ", "", aleatorios))
    singular <- aleatorios[nomes_termo %in% names(v)[is.na(v) | v < TOL_VARIANCIA]]
    if (!length(singular) && !is.finite(as.numeric(logLik(fit)))) {
      singular <- aleatorios[!(nomes_termo %in% names(v))]           # sem informação de variância
    }
    if (!length(singular)) break
    removidos <- c(removidos, singular)
    aleatorios <- setdiff(aleatorios, singular)
  }
  if (!is.finite(as.numeric(logLik(fit)))) stop("ajuste sem log-verossimilhança finita mesmo após retirar efeitos singulares")
  list(fit = fit, aleatorios = aleatorios, removidos = removidos)
}

saida_glm <- function(fit) {
  co <- summary(fit)$coefficients
  out <- list(formula = deparse(formula(fit), width.cutoff = 500L), familia = "poisson", motor = "stats::glm",
              fixos = list(nome = rownames(co), estimativa = unname(co[, 1]), ep = unname(co[, 2])),
              vcov_fixos = unname(as.matrix(vcov(fit))), variancias = list(), ic_variancias = NULL,
              aleatorios = list(), loglik = as.numeric(logLik(fit)), aic = AIC(fit),
              gl = attr(logLik(fit), "df"), convergiu = isTRUE(fit$converged),
              deviance = deviance(fit), deviance_nula = fit$null.deviance,
              dispersao_pearson = sum(residuals(fit, type = "pearson")^2) / fit$df.residual)
  if (isTRUE(spec$anova)) {
    a <- anova(fit, test = "Chisq")
    out$anova <- list(termo = rownames(a), gl = a$Df, deviance = a$Deviance, gl_residual = a[["Resid. Df"]],
                      deviance_residual = a[["Resid. Dev"]], p_valor = a[["Pr(>Chi)"]])
  }
  out
}

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

usar_glm <- !length(spec$aleatorios) && spec$familia == "poisson"
if (usar_glm) {
  # Sem efeitos aleatórios: GLM do R base (stats::glm), o motor do framework da SOA (seção 4.1), com a ANOVA
  # sequencial (deviance explicada por variável, na ordem da fórmula).
  completo <- glm(formula_de(spec$fixos, character(0)), data = dados, family = poisson,
                  control = glm.control(maxit = 100))
  aj <- list(fit = completo, aleatorios = character(0), removidos = character(0))
} else {
  aj <- ajustar(spec$fixos, spec$aleatorios)
  completo <- aj$fit
}
out <- list(modelo = if (usar_glm) saida_glm(completo) else saida_ajuste(completo), testes = list(),
            aleatorios_removidos = aj$removidos)
ll <- function(f) as.numeric(logLik(f))
gl <- function(f) attr(logLik(f), "df")
# razão de verossimilhança: retira um termo por vez do modelo completo (mesma estrutura aleatória final)
for (termo in spec$testar_fixos) {
  red <- if (usar_glm) update(completo, formula_de(setdiff(spec$fixos, termo), character(0))) else
    ajustar_bruto(setdiff(spec$fixos, termo), aj$aleatorios)
  lr <- 2 * (ll(completo) - ll(red))
  out$testes[[length(out$testes) + 1]] <- list(termo = termo, tipo = "fixo", lr = lr, gl = gl(completo) - gl(red),
                                                p_valor = pchisq(max(lr, 0), gl(completo) - gl(red), lower.tail = FALSE))
}
for (termo in spec$testar_aleatorios) {
  if (termo %in% aj$removidos) {     # variância nula: retirado no ajuste, sem evidência de heterogeneidade
    out$testes[[length(out$testes) + 1]] <- list(termo = termo, tipo = "aleatorio", lr = 0, gl = 1, p_valor = 1)
    next
  }
  red <- ajustar_bruto(spec$fixos, setdiff(aj$aleatorios, termo))
  lr <- 2 * (ll(completo) - ll(red))
  # variância na fronteira sob H0: mistura 50:50 de qui-quadrado(0) e qui-quadrado(1)
  out$testes[[length(out$testes) + 1]] <- list(termo = termo, tipo = "aleatorio", lr = lr, gl = 1,
                                                p_valor = 0.5 * pchisq(max(lr, 0), 1, lower.tail = FALSE))
}
writeLines(toJSON(out, auto_unbox = TRUE, digits = NA, null = "null", na = "null"), spec$saida)
