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
| E a taxa comercial, com carregamentos? | Cotador (próximo) |
| Quanto reajustar na renovação, e quanto confiar na experiência do grupo? | Etapa 3 |
| Qual a chance de a sinistralidade passar da meta no próximo ano? | Etapa 4 |
| O método novo teria acertado mais que a regra contratual no passado? | Etapa 5 (backtest) |

## Status

- [x] **Etapa 1: dados.** Leitura (Excel, CSV ou parquet) com mapeamento de colunas; 26 regras de
  qualidade com log auditável; exposição em vidas-ano e capital-ano; óbitos e valores esperados; número
  efetivo de óbitos ponderado por capital; conciliação de prêmio; gerador sintético.
- [x] **Etapa 2: taxa manual.** Modelo de A/E sobre a BR-EMSmt 2026 com efeito aleatório de
  estipulante (Poisson-Gama, verossimilhança exata), comparado com GLM, Binomial Negativa e ridge;
  seleção de fatores por razão de verossimilhança; estudo de simulação com 12 carteiras; tabela de
  fatores versionada em `artefatos/fatores_manual.yaml`; função de taxa pura para um censo de vidas.
- [ ] Etapa 3: renovação (Bühlmann-Straub, pooling de sinistros grandes, IBNR, regra por faixa)
- [ ] Etapa 4: camada bayesiana (PyMC)
- [ ] Etapa 5: validação e backtest
- [ ] Cotador, relatório Quarto e checklist de governança

## Como rodar

```bash
pip install -e ".[dev]"
pytest                                   # 72 testes (1 só roda com o arquivo oficial da SUSEP presente)
jupyter lab                              # rode notebooks/01_dados e depois 02_taxa_manual
python scripts/estudo_simulacao_manual.py   # opcional (~6 min): refaz o estudo de simulação
```

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
- **Como um fator entra na tabela de preço?** Por teste de razão de verossimilhança (5%). Idade e sexo
  já estão na tábua; o teste mede só o desvio adicional.
- **O pipeline corrige dados sozinho?** Não. Toda exclusão ou sinalização vai para o log de exceções,
  com a regra e a linha de origem.

## Estrutura

```
config/      parâmetros versionados (tábua, regras de validação, mapeamento, hipóteses do sintético)
src/vgpricing/  tabua, io, validacao, exposicao, sintetico, template, manual
artefatos/   tabela de fatores da taxa manual (lida pelo cotador)
resultados/  estudo de simulação da taxa manual
scripts/     estudos que demoram mais que um notebook
notebooks/   narrativa de cada etapa (só chama funções testadas)
tests/       pytest: casos calculados à mão + prova de detecção da sujeira injetada
templates/   Excel de entrada
data/        external/ (tábua oficial, fora do git), raw/ (dados reais, fora do git), synthetic/
```

## Governança e LGPD

Dados reais nunca entram no repositório (`.gitignore`). Identificadores entram pseudonimizados
(`vgpricing.io.hash_id`, com sal guardado fora do código). A versão da tábua é travada por hash.
