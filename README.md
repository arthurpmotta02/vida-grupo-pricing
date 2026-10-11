# vida-grupo-pricing

Ferramenta de cotação e renovação de **Vida em Grupo** (com extensões previstas para Prestamista e AP),
construída do jeito que uma área de precificação trabalha: taxa manual a partir da tábua, ajuste pela
experiência do próprio grupo com credibilidade, e uma camada bayesiana usada só onde ela muda a decisão
(incerteza da renovação, probabilidade de estourar a meta de sinistralidade).

> Todos os dados deste repositório são **sintéticos**. Não há microdados públicos de vida em grupo no
> Brasil; o gerador cria uma carteira realista com "verdade conhecida" para testar cada etapa. O código lê
> dados reais sem reescrita: basta ajustar o mapeamento de colunas em `config/colunas.yaml`.

## Para quem é e o que entrega

| Pergunta de negócio | Onde |
|---|---|
| A base de vidas e sinistros está confiável? O que precisa voltar para o cadastro? | Etapa 1 (pronta) |
| Qual a taxa pura por mil para um grupo novo? | Etapa 2 (pronta) |
| Quanto reajustar na renovação? | Etapa 3 (pronta) |
| E a taxa comercial, com carregamentos? | Cotador (próximo) |
| Qual a chance de a sinistralidade passar da meta no próximo ano? | Etapa 4 |
| O método novo teria acertado mais que a regra contratual no passado? | Etapa 5 (backtest) |

## Status

- [x] **Etapa 1: dados.** Leitura (Excel, CSV ou parquet) com mapeamento de colunas; 27 regras de
  qualidade com log auditável; exposição em vidas-ano e capital-ano; óbitos e valores esperados; número
  efetivo de óbitos ponderado por capital; conciliação de prêmio; gerador sintético.
- [x] **Etapa 2: taxa manual.** GLMM de Poisson estimado no R com **glmmTMB** (conferido com lme4):
  offset dos óbitos esperados pela BR-EMSmt 2026, log do capital, seção CNAE como efeito fixo, divisão
  CNAE e estipulante como efeitos aleatórios (estrutura de Ohlsson para fatores com muitos níveis).
  Idade, sexo e porte só entram com evidência (razão de verossimilhança com correção de Holm).
  Tabela de fatores versionada em `artefatos/fatores_manual.yaml` e taxa pura para um censo de vidas.
- [x] **Etapa 3: renovação.** Pooling formal com encargo; A/E contra a taxa manual em valor limitado;
  IBNR por chain ladder mensal (pacote chainladder) com alocação Bornhuetter-Ferguson; credibilidade de
  mercado (raiz, vidas-ano) e Bühlmann-Straub com número efetivo de óbitos (conferido com `actuar::cm`);
  reajuste técnico com piso e teto; comparação com a regra contratual; backtest fora do tempo com
  deviance e intervalo por bootstrap pareado.
- [ ] Etapa 4: diagnóstico com machine learning (LightGBM Poisson sobre o resíduo da taxa manual, SHAP).
  Só aponta interações candidatas; não entra na tarifa.
- [ ] Etapa 5: camada bayesiana (mesmo modelo com prioris; propaga a incerteza da variância entre grupos).
- [ ] Etapa 6: validação por backtest fora do tempo e cenários em que as hipóteses do modelo falham.
- [ ] Etapa 7: cotador (carregamentos, retenção, concentração). Etapa 8: relatório Quarto e governança.

## Como rodar

```bash
pip install -e ".[dev]"
Rscript r/instalar_pacotes.R             # R >= 4.2 com glmmTMB, lme4, jsonlite, actuar
pytest                                   # testes do GLMM e da tábua oficial são pulados se faltar R/arquivo
jupyter lab                              # rode os notebooks na ordem: 01, 02, 03
```

**Por que R.** glmmTMB e lme4 são as implementações de referência de modelos mistos, e a literatura
atuarial de credibilidade com modelos mistos usa R. O Python organiza os dados e chama o R por
`Rscript` (arquivos CSV/JSON, sem rpy2). O cotador só lê a tabela de fatores: **cotar não exige R**,
estimar exige. Em Ubuntu, os pacotes existem no apt (`r-cran-glmmtmb`, `r-cran-lme4`, `r-cran-actuar`).

**Tábua.** O pipeline usa a BR-EMSmt 2026 oficial, lida direto do arquivo publicado pela SUSEP e
travada por hash. Na primeira execução, se o arquivo não estiver em `data/external/`, o código baixa
sozinho do site da SUSEP e confere o hash (`tabua.baixar_automaticamente` no config). Se o download
falhar (rede corporativa, link alterado), baixe manualmente; detalhes em `data/external/README.md`.
Sem o arquivo, roda com uma curva **placeholder** (não é BR-EMS), usada pelos testes, e avisa.

**Dados reais.** Preencha `templates/entrada_vida_grupo.xlsx` (o Excel é só entrada; nenhuma conta é
feita nele) ou aponte para uma pasta com parquet/CSV extraídos do banco. Ajuste `config/colunas.yaml`
se os nomes de coluna forem outros.

## Decisões que um auditor vai perguntar

- **Por que exposição em segmentos, e não foto mensal?** A foto mensal perde entradas e saídas no meio
  do mês e infla ou reduz a exposição. Há um conversor `faturamento_para_segmentos` para bases mensais.
- **Por que n efetivo, e não número de vidas, como peso de credibilidade?** Quando o A/E é medido em
  R$, um grupo com poucas vidas de capital alto oscila muito mais do que o número de vidas sugere.
  O n efetivo, (Σ C·q)² / Σ C²·q, mede essa oscilação.
- **O que acontece com afastados e aposentados por invalidez?** Saem da base de precificação (risco de
  outra natureza), mas continuam na conciliação de prêmio, porque são faturados.
- **Por que a taxa manual é estimada em número de óbitos, e não em R$?** Em valor, poucos sinistros de
  capital alto dominariam os fatores. A visão em valor entra na renovação, com pooling.
- **Por que um modelo com efeito de grupo, e não um GLM comum?** No estudo de simulação, o GLM com todos
  os fatores previu grupos novos pior que a própria tábua, e o intervalo de 95% dele acertou só ~80% das
  vezes. O modelo com efeito de grupo e fatores selecionados previu melhor e teve cobertura de ~94%.
  Além disso, ele estima o k de credibilidade usado na renovação.
- **Por que um modelo misto, e não um GLM comum?** Células do mesmo grupo não são independentes. O GLM
  trata cada uma como informação nova e produz intervalos estreitos demais. O modelo misto separa o
  efeito da atividade do efeito do grupo; credibilidade e modelos mistos são a mesma teoria (CAS
  Monograph 14). Na previsão de grupos novos, GLM e modelo misto empatam; a diferença está na incerteza.
- **Como a atividade entra?** Seção CNAE como fator fixo e divisão como efeito aleatório dentro da seção:
  cada divisão recebe credibilidade em torno da sua seção (Ohlsson, 2008). Seções com pouca exposição
  vão para "OUTRAS".
- **Como um fator entra na tabela de preço?** Atividade e capital por decisão de negócio, tomada antes de
  olhar os dados. Idade, sexo e porte só com razão de verossimilhança, corrigida por Holm para testes
  múltiplos (sem a correção, o porte entraria por acaso).
- **Grupo novo tem fator 1?** Não: com efeito Normal na escala log, o fator esperado de um grupo
  desconhecido é exp(σ²/2). A tabela já inclui essa correção.
- **Quanto confiar na variância entre grupos?** Pouco, com o volume da carteira sintética (mediana de 3
  óbitos por grupo): o intervalo é largo e aparece no notebook. A camada bayesiana (Etapa 5) propaga
  essa incerteza para o preço.
- **Por que o A/E da renovação é medido contra a taxa manual?** Porque a credibilidade mistura a
  experiência do grupo com a manual; medir contra a tábua misturaria o efeito da atividade com o do grupo.
- **Qual fórmula de credibilidade?** A de mercado (flutuação limitada em vidas-ano) e a de Bühlmann-Straub
  são calculadas lado a lado. A escolha é por backtest fora do tempo, com intervalo de confiança: com pouco
  volume, um único ano não separa os métodos, e isso aparece no notebook.
- **O pipeline corrige dados sozinho?**## Estrutura

```
config/      parâmetros versionados (tábua, regras de validação, mapeamento, hipóteses do sintético)
src/vgpricing/  tabua, io, validacao, exposicao, sintetico, template, glmm, manual, ibnr, renovacao, backtest
r/           scripts R (GLMM, Bühlmann-Straub do actuar, instalação de pacotes)
artefatos/   tabela de fatores da taxa manual (lida pelo cotador)
notebooks/   narrativa de cada etapa (só chama funções testadas)
tests/       pytest: casos calculados à mão + prova de detecção da sujeira injetada
templates/   Excel de entrada
data/        external/ (tábua oficial, fora do git), raw/ (dados reais, fora do git), synthetic/
```

## Governança e LGPD

Dados reais nunca entram no repositório (`.gitignore`). Identificadores entram pseudonimizados
(`vgpricing.io.hash_id`, com sal guardado fora do código). A versão da tábua é travada por hash.
