# GLM de Poisson com penalização elastic net (glmnet), com offset.
#   Rscript r/ajustar_glmnet.R especificacao.json
# - O offset carrega o que já se sabe (log dos esperados pela tábua e, opcionalmente, a "priori" de um estudo
#   de mercado); os coeficientes penalizados medem o desvio em relação a ele. Penalização ridge = priori
#   normal centrada no offset (estimativa MAP); o lambda, escolhido por validação cruzada, faz o papel do k
#   da credibilidade (SOA, 2024, seção 4.3.1; CAS Monograph 5, seção 10.5).
# - Categóricas em codificação completa (um coeficiente por nível). `standardize` segue o padrão do glmnet
#   (TRUE), como no framework da SOA, salvo se a especificação pedir outro valor.
suppressMessages({library(glmnet); library(Matrix); library(jsonlite)})
spec <- fromJSON(commandArgs(trailingOnly = TRUE)[1], simplifyVector = TRUE)

ler <- function(p) read.csv(p, stringsAsFactors = FALSE, colClasses = "character")
treino <- ler(spec$treino)
outros <- lapply(spec$prever, ler)
todos <- do.call(rbind, c(list(treino), unname(outros)))
n_tr <- nrow(treino)
for (v in spec$numericas) todos[[v]] <- as.numeric(todos[[v]])
for (v in spec$categoricas) todos[[v]] <- factor(todos[[v]])
todos$N <- as.numeric(todos$N)
todos$offset <- as.numeric(todos[[spec$offset]])

f <- as.formula(paste("~", paste(spec$termos, collapse = " + ")))
cats <- intersect(as.character(unlist(spec$categoricas)), all.vars(f))
contr <- if (length(cats)) lapply(todos[cats], contrasts, contrasts = FALSE) else NULL
# matriz esparsa: com interações e centenas de milhares de linhas, a densa não caberia na memória
X <- sparse.model.matrix(f, data = todos, contrasts.arg = contr)[, -1, drop = FALSE]
if (ncol(X) < 2) X <- cbind(X, .nula = 0)   # o glmnet exige 2+ colunas; a coluna nula fica com coeficiente 0
Xtr <- X[seq_len(n_tr), , drop = FALSE]
set.seed(spec$semente)
args_cv <- list(x = Xtr, y = todos$N[seq_len(n_tr)], family = "poisson", offset = todos$offset[seq_len(n_tr)],
                alpha = spec$alpha, standardize = if (is.null(spec$padronizar)) TRUE else isTRUE(spec$padronizar))
if (!is.null(spec$foldid)) {
  args_cv$foldid <- as.integer(treino[[spec$foldid]])
} else {
  args_cv$nfolds <- spec$nfolds
}
cv <- do.call(cv.glmnet, args_cv)
lam <- cv[[spec$lambda]]
b <- as.matrix(coef(cv, s = lam))

# previsões (média) para cada conjunto pedido, gravadas em CSV
inicio <- n_tr
saidas <- list()
for (nome in names(spec$prever)) {
  n <- nrow(outros[[nome]])
  idx <- inicio + seq_len(n)
  eta <- as.numeric(b[1, 1] + X[idx, , drop = FALSE] %*% b[-1, 1]) + todos$offset[idx]
  arq <- file.path(spec$pasta_saida, paste0("prev_", nome, ".csv"))
  write.csv(data.frame(mu = exp(eta)), arq, row.names = FALSE)
  saidas[[nome]] <- arq
  inicio <- inicio + n
}
eta_tr <- as.numeric(b[1, 1] + Xtr %*% b[-1, 1]) + todos$offset[seq_len(n_tr)]
write.csv(data.frame(mu = exp(eta_tr)), file.path(spec$pasta_saida, "prev_treino.csv"), row.names = FALSE)

out <- list(lambda = lam, lambda_min = cv$lambda.min, lambda_1se = cv$lambda.1se,
            caminho = list(lambda = cv$lambda, deviance_cv = cv$cvm, ep = cv$cvsd, nao_nulos = cv$nzero),
            coeficientes = list(nome = rownames(b), valor = unname(b[, 1])),
            previsoes = saidas)
writeLines(toJSON(out, auto_unbox = TRUE, digits = NA), spec$saida)
