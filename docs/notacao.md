# Notação e teoria

Guia de referência dos símbolos e métodos do projeto, na ordem do pipeline. Cada seção diz **o que é**, **a fórmula**
e **onde aparece** (notebook e módulo). A seção 11 compara os métodos (vantagens, desvantagens, quando usar).

Para gerar o PDF (LaTeX, via Quarto) a partir deste mesmo arquivo: `quarto render docs/notacao.md --to pdf`.

## Símbolos gerais

| Símbolo | Significado |
|---|---|
| $x$ | idade (completa) da vida |
| $q_x$ | probabilidade de morte em um ano na idade $x$ (tábua BR-EMSmt 2026, por sexo) |
| $t$ | fração do ano em que a vida esteve exposta (dias ÷ 365,25) |
| $C$ | capital segurado da vida |
| $C_{ref}$ | capital de referência (R\$ 50 mil): só normaliza o fator de capital |
| $c$ | célula de modelagem (grupo × faixa etária × sexo × faixa de capital) ou célula da SOA |
| $g, i$ | grupo (estipulante); na credibilidade, $i$ indexa grupos e $j$ períodos |
| $N$ | óbitos observados (contagem) |
| $E$ | óbitos esperados pela tábua |
| $\mu$ | óbitos esperados por um modelo |
| $A$ | observado (contagem ou valor), com IBNR quando indicado |
| $P$ | nível de pooling do grupo |
| $Z$ | credibilidade (peso da experiência própria) |

## 1. Exposição e A/E (01, 02 — `exposicao`, `estudo`)

$$
E = \sum_{\text{vidas}} q_x\,t, \qquad E^{(v)} = \sum_{\text{vidas}} q_x\,t\,C, \qquad
\frac{A}{E} = \frac{N}{E}.
$$

- **Vidas-ano** $=\sum t$; **capital-ano** $=\sum t\,C$.
- **Número efetivo de óbitos.** Com capitais diferentes, a variância do valor observado é $\sum C^2 q t$ (Poisson
  composta), e o A/E em valor tem a precisão de
  $$n_{ef} = \frac{\left(\sum C q t\right)^2}{\sum C^2 q t} \le E,$$
  com igualdade quando todos os capitais são iguais. É o peso da credibilidade em valor.
- **Intervalo exato de Poisson (Garwood)** para a média de uma contagem $N$, nível $1-\alpha$:
  $$\left[\tfrac{1}{2}\chi^2_{\alpha/2}(2N),\; \tfrac{1}{2}\chi^2_{1-\alpha/2}(2N+2)\right],$$
  dividido por $E$ para o A/E. Aproximação: $\text{A/E} \pm 1{,}96/\sqrt{N}$. Com 30 óbitos, cerca de ±36%: daí o
  mínimo de 30 esperados para um nível ter fator próprio.

## 2. GLM de Poisson com offset (03a, 03b — `glmm`, R `stats::glm`)

$$
N_c \sim \text{Poisson}(\mu_c), \qquad \log \mu_c = \underbrace{\log E_c}_{\text{offset}} + \beta_0 + \sum_k \beta_k x_{ck}.
$$

O offset fixa a tábua: cada $e^{\beta_k}$ é um fator multiplicativo sobre ela.

- **Deviance:** $D = 2\sum_c \left[N_c \log\frac{N_c}{\mu_c} - (N_c - \mu_c)\right]$ (termo com $N_c = 0$ vale $\mu_c$).
  Menor é melhor; é a medida de erro adequada a contagens de eventos raros.
- **AIC** $= 2p - 2\log L$, com $p$ parâmetros. Só compara modelos ajustados aos mesmos dados.
- **ANOVA sequencial:** deviance explicada por cada variável, entrando na ordem da fórmula (tabela da SOA, seção 4.1).
- **Dispersão de Pearson** $=\sum (N-\mu)^2/\mu \,/\,(n-p)$: perto de 1 indica que a Poisson descreve a variância.
- **Razão de verossimilhança:** $LR = 2(\log L_1 - \log L_0) \sim \chi^2_{gl}$. Para testar uma variância de efeito
  aleatório (fronteira do espaço de parâmetros), $p = \tfrac12 P(\chi^2_1 > LR)$.
- **Correção de Holm** para $m$ testes: ordenam-se os p-valores e
  $p^{adj}_{(k)} = \max_{j \le k} \min\{1, (m - j + 1)\,p_{(j)}\}$. Controla a chance de algum falso positivo.

## 3. Modelo misto (GLMM) (03b — R `glmmTMB`)

$$
\log \mu_c = \log E_c + \mathbf{x}_c'\boldsymbol\beta + u_{g(c)} + v_{d(c)}, \qquad u_g \sim N(0, \sigma^2_{est}),\; v_d \sim N(0, \sigma^2_{div}).
$$

- O efeito do estipulante $u_g$ captura a mortalidade própria do grupo; a divisão CNAE $v_d$, dentro da seção, dá
  credibilidade a cada divisão em torno da sua seção (Ohlsson, 2008).
- **Previsão de grupo novo** (sem $u_g$ observado): $\mathrm{E}[e^{u}] = e^{\sigma^2/2}$, não 1.
- **Ajuste singular:** variância estimada abaixo de $10^{-8}$; o efeito é retirado e o modelo reajustado.

## 4. Penalização: elastic net, ridge e "priori" (03a, 03b — `elasticnet`, R `glmnet`)

$$
\hat{\boldsymbol\beta} = \arg\min_{\boldsymbol\beta}\; -\frac{1}{n}\log L(\boldsymbol\beta) + \lambda\left[\frac{1-\alpha}{2}\lVert\boldsymbol\beta\rVert_2^2 + \alpha\lVert\boldsymbol\beta\rVert_1\right].
$$

- $\alpha = 1$: LASSO (zera coeficientes); $\alpha = 0$: ridge (encolhe todos). $\lambda$ por validação cruzada
  (`lambda.min`: o de menor erro).
- **Ridge é uma priori normal.** Maximizar a verossimilhança penalizada por $\lambda\lVert\beta\rVert^2$ é achar a
  moda da posteriori com $\beta \sim N(0, \tau^2)$, $\lambda \propto 1/\tau^2$ (MAP). LASSO corresponde a uma priori
  de Laplace.
- **Priori de um estudo de mercado.** Colocando a forma do estudo no offset,
  $$\log \mu = \log E + \beta^{SOA}\log\tfrac{C}{C_{ref}} + \log f^{SOA}_{s} + b_0 + \delta \log\tfrac{C}{C_{ref}} + \gamma_s,$$
  a penalização encolhe $(\delta, \gamma_s)$ para zero, isto é, para o estudo. É a mesma lógica da credibilidade:
  $\lambda$ alto → fica no estudo; dados abundantes → segue a carteira (SOA, 2024, seção 4.3.1, liga $\lambda$ ao $k$
  de Bühlmann).

## 4b. AGLM (03a, 03b — `aglm`, R `glmnet`)

GLM regularizado com codificações que capturam não linearidade sem perder a forma de tabela (Fujita, Tanaka, Kondo e
Iwasawa, 2020).

- **Variáveis O** (dummies ordinais): para uma variável com níveis ordenados $1, \dots, m$,
  $d^O_j(x) = \mathbb{1}\{x < j\}$, $j = 2, \dots, m$. O coeficiente $\beta_j$ é o **salto** entre os níveis
  $j-1$ e $j$; com LASSO, $\beta_j = 0$ funde as duas faixas. É equivalente ao *fused lasso*
  ($\lambda \sum_j |\gamma_j - \gamma_{j-1}|$ sobre os efeitos de nível $\gamma$).
- **Variáveis L**: com nós $b_1 < \dots < b_{m-1}$, $l_j(x) = |x - b_j|$, mais o próprio $x$. A contribuição
  $\sum_j \beta_j l_j(x)$ é uma curva **linear por partes e contínua**; $\beta_j$ é a mudança de inclinação no nó $b_j$.
- No projeto: capital em variáveis L (9 nós nos quantis do log), faixa etária e porte em O, seção e sexo em dummies
  comuns, LASSO, e a forma da SOA no offset. A tabela de fatores guarda os nós e os coeficientes.

## 4c. GAM (03a, 03b — `gam`, R `mgcv`)

$$\log \mu = \log E + \beta_0 + f(\log C) + \mathbf{x}'\boldsymbol\beta, \qquad f(z) = \sum_k \theta_k B_k(z),$$

com $B_k$ uma base de splines e uma penalização de rugosidade $\lambda \int f''(z)^2\,dz$ escolhida por REML. Os
**graus de liberdade efetivos** (edf) medem a complexidade usada: edf ≈ 1 é uma reta (no log, uma elasticidade
constante). A penalização é uma priori gaussiana sobre a rugosidade (interpretação bayesiana de Wood, 2017).

## 5. LightGBM, SHAP e interações (03a, 03b — `diagnostico`)

- **Gradient boosting** com perda Poisson e valor inicial $\log E$: soma árvores que corrigem o log da taxa.
- **SHAP:** decomposição aditiva da previsão, $\log\mu(\mathbf{x}) = \log\mu_0 + \sum_j \phi_j(\mathbf{x})$;
  importância = média de $|\phi_j|$.
- **Tabela de interações (estilo EIX):** para cada divisão de uma árvore cujo nó pai divide por outra variável, o par
  (pai, filho) acumula o ganho do filho (*sumGain*), as observações do filho (*sumCover*) e a frequência.

## 6. Credibilidade (04, 05 — `renovacao`, R `actuar`)

**Flutuação limitada (prática de mercado):** $Z = \min\left(1, \sqrt{n/n_{full}}\right)$, com $n$ em vidas-ano,
$n_{full} = 20\,000$ e $Z = 0$ abaixo de 500.

**Bühlmann-Straub.** Grupo $i$, período $j$, razão observada $X_{ij}$ (aqui, A/E em valor contra a taxa manual) e
peso $w_{ij}$ (aqui, $n_{ef}$). Com $w_i = \sum_j w_{ij}$, $\bar X_i = \sum_j w_{ij}X_{ij}/w_i$ e
$\bar X = \sum_i w_i \bar X_i / w$:

$$
\hat s^2 = \frac{\sum_i\sum_j w_{ij}(X_{ij} - \bar X_i)^2}{\sum_i (n_i - 1)}, \qquad
\hat a = \frac{\sum_i w_i(\bar X_i - \bar X)^2 - (I - 1)\hat s^2}{w - \sum_i w_i^2/w},
$$

$$
k = \frac{\hat s^2}{\hat a}, \qquad Z_i = \frac{w_i}{w_i + k}, \qquad \hat X_i = Z_i \bar X_i + (1 - Z_i)\,\mu .
$$

$s^2$ é a variância dentro do grupo (ruído) e $a$ a variância entre grupos (sinal). Se $\hat a \le 0$, $Z = 0$.
Na renovação, o complemento $\mu$ é a própria taxa manual (A/E = 1). É a melhor previsão linear (Bayes linear) do
risco do grupo.

## 7. Pooling e IBNR (04 — `renovacao`, `ibnr`)

- **Nível de pooling:** $P = \min\{\text{retenção}, \max(\text{piso}, m\,\bar C)\}$, com $\bar C$ o capital médio do
  grupo. Na experiência entra $\min(C, P)$; o excesso esperado vira **encargo** cobrado de todos:
  $\sum q f \max(C - P, 0)$.
- **Chain ladder mensal** (carteira): fatores de desenvolvimento $f_j = \sum_i S_{i,j+1} / \sum_i S_{i,j}$ (volume),
  $\text{CDF}_j = \prod_{k \ge j} f_k$, proporção avisada $p_j = 1/\text{CDF}_j$.
- **Bornhuetter-Ferguson por grupo:** $\text{IBNR} = \sum_{\text{meses}} E^{\text{apriori}}_m\,(1 - p_{a(m)})$, com
  $a(m)$ a idade do mês de ocorrência na avaliação. Estável para grupos com poucos sinistros.
- **Fator de desenvolvimento da taxa manual:** média de $1/p_{a}$ sobre os sinistros avisados da experiência
  (cada um representa $1/p_a$ sinistros finais).

## 8. Reajuste e cotação (04, 06 — `renovacao`, `cotador`)

$$
\text{prêmio puro} = \sum q\,f_{\text{manual}}\min(C,P)\cdot\text{fator} + \text{encargo}, \qquad
\text{taxa comercial} = \text{taxa pura}\cdot\frac{1 + \text{margem}}{1 - \sum \text{carregamentos}},
$$

$$
\text{reajuste} = \operatorname{clip}\left(\frac{\text{taxa comercial}}{\text{taxa vigente}} - 1,\; \text{piso},\; \text{teto}\right).
$$

Na cotação de grupo novo, fator = 1. $f_{\text{manual}}$ = A/E base × fator de seção × fator de divisão ×
$(C/C_{ref})^{\beta_{cap}}$.

## 9. Validação (03a, 03b, 05 — `avaliacao`, `backtest`)

- **Teste fora do tempo:** ajusta-se com dados até uma data e avalia-se no período seguinte; o teste só é usado no
  fim (CAS, 4.3).
- **Bootstrap pareado:** reamostram-se as unidades (estipulantes, seguradoras) com reposição e soma-se a diferença de
  deviance entre dois modelos; o IC de 95% são os percentis 2,5 e 97,5. "Significativa" = IC sem o zero.
- **Gráfico de quantis:** ordena-se pela relatividade prevista $\mu/E$, faixas de igual $E$; previsto e observado
  por faixa (CAS 7.2.1).
- **Double lift:** ordena-se pela razão entre dois modelos; vence o que acompanha o observado (CAS 7.2.2).
- **Lorenz e Gini:** ordenando do menor para o maior risco previsto, $x$ = parcela acumulada de $E$, $y$ = parcela
  acumulada de $N$; Gini = 2 × área entre a diagonal e a curva (CAS 7.2.4).
- **A/M:** observado ÷ modelo por nível; desvios acima de ±5% a ±10% indicam ajuste ruim (SOA 2.4.5); aqui também o IC
  exato.
- **Gráfico de sinistralidade:** ordena-se pelo custo previsto ÷ prêmio atual; sinistralidade observada por faixa
  (CAS 7.2.3).
- **Estabilidade:** bootstrap por estipulante dos fatores da tarifa (CAS 6.4).
- **Regra de decisão:** menor deviance no teste; empatados (IC contém zero) resolvidos por uma ordem de preferência
  fixada antes.

## 10. Monitoramento (07 — `monitoramento`)

- **Limites da carteira:** sob "a taxa está certa", $N \sim \text{Poisson}(E)$; limites do A/E =
  quantis de $\text{Poisson}(E)$ ÷ $E$, para 95% e 99,8% (funnel plot, Spiegelhalter 2005).
- **Limites dos grupos:** com mortalidade própria $u \sim \text{Gama}$ de média 1 e variância $\tau^2$,
  $N \sim \text{Binomial Negativa}$ com média $E$ e variância $E + \tau^2 E^2$. Estimador de momentos:
  $\hat\tau^2 = \max\left\{0,\; \sum\left[(N - E)^2 - N\right] / \sum E^2\right\}$.
- **IBNR nos meses recentes:** $N^{aj} = N/p_a$; os meses com $a \le 3$ ficam fora do A/E móvel.

## 10b. Dados faltantes (01 — `exposicao`, `validacao`, `manual`)

- **Sexo ausente:** com $p$ a proporção de homens do perfil, o esperado é
  $$\mathrm{E}[q] = p\,q^M_x + (1 - p)\,q^F_x,$$
  **exato** para o número esperado de óbitos, que é linear em $p$. $p$ vem do nível mais fino com dados: grupo e faixa
  de 10 anos, grupo, seção CNAE e faixa, carteira; para a cotação de um grupo sem sexo, a tabela de fatores guarda a
  proporção da carteira por seção e faixa etária. A média 50/50 só é certa se $p = 0{,}5$.
- **Ausência informativa** (*missing not at random*): se a falta de sexo se concentra em certos grupos ou atividades, a
  ausência carrega informação. Por isso "ND" é um nível próprio nas células, testado como fator.
- **Nascimento ausente:** imputação pela idade de entrada mediana do grupo, com indicador. Excluir a vida retira
  exposição e enviesa o A/E; o indicador permite medir a sensibilidade.
- **Gradient boosting** (LightGBM) aceita valores ausentes nativamente: aprende para que lado da árvore mandá-los. É
  cômodo, mas implícito: o modelo pode usar "ausente" como substituto de outra coisa, sem que isso apareça. Além disso,
  o sexo entra pela tábua (no offset), e o esperado precisa dele de qualquer forma. A ASOP 23 pede que o tratamento
  seja explícito e documentado.

## 11. Comparação dos métodos

| Método | Vantagens | Desvantagens | Quando usar |
|---|---|---|---|
| **GLM de Poisson com offset** | interpretável fator a fator; erro-padrão; padrão regulatório e de mercado; rápido | forma fixada pelo modelador; sem encolhimento para níveis raros | tarifa, quando há volume para os fatores |
| **GLMM (efeitos aleatórios)** | credibilidade automática para níveis e grupos; separa ruído de heterogeneidade | efeitos fixos são condicionais ao grupo (como taxa de grupo novo, podem prever pior); ajuste singular com pouca variância | estimar a variância entre grupos; atividades com muitos níveis |
| **Ridge / elastic net** | encolhe níveis raros; aceita priori via offset; LASSO seleciona variáveis | sem erro-padrão (estabilidade por bootstrap); lambda depende da validação cruzada | pouca experiência; muitos níveis; incorporar estudo de mercado |
| **AGLM** | não linearidade e fusão de faixas sem escolher bins à mão; continua sendo tabela | curvas podem não ser monótonas; mais parâmetros para explicar | variáveis contínuas ou ordenadas com forma desconhecida |
| **GAM** | forma suave e estimada; edf mostra se a forma linear basta | curva precisa ser tabulada para tarifa; menos direto de explicar | diagnóstico da forma de variáveis contínuas |
| **LightGBM** | captura interações e não linearidades sem especificar; aceita ausentes | não explicável fator a fator; exige regularização forte com eventos raros | referência de desempenho; busca de estrutura faltante |
| **Credibilidade de mercado (raiz)** | simples, conhecida no mercado | limiares arbitrários; ignora concentração de capital | comparação (*champion*) |
| **Bühlmann-Straub** | k estimado nos dados; peso pelo número efetivo; melhor previsão linear | supõe estrutura estável entre períodos; k incerto com poucos grupos | renovação |
| **Chain ladder + Bornhuetter-Ferguson** | padrão de mercado; BF estável para grupos pequenos | supõe padrão de aviso estável | IBNR da carteira e alocação por grupo |
