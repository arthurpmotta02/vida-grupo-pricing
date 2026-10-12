# Dados públicos (não versionados)

Gerados por `scripts/preparar_dados_publicos.py`, na máquina do usuário:

| Arquivo | Origem | Conteúdo |
|---|---|---|
| `soa_basic_life_morte_agregado.csv.gz` | SOA, 2016 Group Term Life Mortality Study, tabela "Basic Life Death and Dis AE Ratios" (.xlsb convertido) | células de exposição, óbitos e esperados de morte por atividade, colarinho, faixa de capital, salário, porte, sexo e idade |
| `ses_seguros_pessoas.csv.gz` | SUSEP SES, `Ses_seguros.csv` | prêmios e sinistros mensais por seguradora, ramos 0993, 0977 e 0982 |
| `Ses_cias.csv`, `Ses_ramos.csv` | SUSEP SES | cadastros de seguradoras e ramos |

Cuidados de leitura (tratados em `vgpricing.publicos`):
- O SES usa vírgula decimal (`191319500,08`), e os cadastros trazem códigos com espaços à direita.
- No SES, o sinistro está em `sinistro_retido` até 2013 e em `sinistro_ocorrido` a partir de 2014; só a série a
  partir de 2014 é usada.
- Na SOA, os esperados são as taxas do próprio estudo por idade e sexo: o A/E total é 100%.
- Na SOA, a atividade "Z. Unknown" concentra cerca de 30% dos óbitos e é classificada como colarinho cinza.
