# Tábua BR-EMS (arquivo oficial)

O arquivo oficial da SUSEP não é versionado no git. O que trava a versão usada é o hash SHA-256
registrado em `config/base.yaml`.

## Primeira vez nesta máquina

Não precisa fazer nada: na primeira execução, `obter_tabua()` baixa o arquivo do link em
`config/base.yaml` (`tabua.url`) e confere o hash. Para baixar à mão pelo Python:

```python
from vgpricing.tabua import baixar_tabua_susep
print(baixar_tabua_susep())   # salva em data/external/tabuas_br_ems.xlsx e imprime o SHA-256
```

Se a rede bloquear (proxy corporativo), baixe pelo navegador na página "Tábuas biométricas BR-EMS"
da SUSEP e salve com o nome acima.

O hash impresso deve ser igual a `tabua.sha256` no config. Se for diferente, a SUSEP republicou o
arquivo: confira o que mudou (aba "Vigência" e as abas da versão em uso) antes de atualizar o hash.

## Estrutura do arquivo (observada em out/2026)

- Aba "Vigência" com as datas de cada versão.
- Uma aba por tábua e sexo para 2015, 2021 e 2026 (ex.: `BR-EMSmt-2026-m`); a de 2010 tem as quatro
  tábuas numa aba só.
- O cabeçalho (`Idade | qx | IC 95% inf | IC 95% sup | lx | ex`) muda de linha entre abas, e algumas têm
  colunas vazias intercaladas. O carregador localiza o cabeçalho sozinho.
- O título da aba `BR-EMSmt-2026-m` traz "Morte Masculina" e, logo abaixo, "Sobrevivência Masculina".
  Por isso o carregador não confia no texto do título: confere numericamente que a tábua lida tem
  mortalidade maior que a de sobrevivência (mt > sb) da mesma versão.

## Achados no arquivo oficial (versão 2026), conferidos em out/2026

- **Coluna `lx` defasada em uma idade.** Nas quatro abas 2026 (mt e sb, M e F), 1 − l(x+1)/l(x) bate
  com q_(x+1), não com q_x (diferença de até ~9% entre 20 e 80 anos; com a defasagem, até 0,2%).
- **A coluna `qx` é a correta.** O e_x publicado coincide com o e_x abreviado calculado a partir da
  `qx` (diferença máxima de 0,05 ano, dentro do arredondamento de uma casa) e não com o calculado pela
  `lx`. Ex.: BR-EMSmt-2026-m, e_0 publicado 78,8; pela `qx` 78,82; pela `lx` 77,82.
- Nas versões 2015 e 2021, `qx`, `lx` e `ex` são consistentes entre si.
- **Corcova de acidentes:** q_x masculino cai entre 26 e 31 anos (mt e sb). Padrão demográfico.

Decisão: o cálculo usa apenas a coluna `qx`. A defasagem do `lx` vira aviso a cada carga, e o teste
`test_arquivo_oficial_br_ems_2026` trava esses achados.

## Checagens feitas a cada carga

1. SHA-256 igual ao do config (senão, bloqueia).
2. Idades consecutivas, 0 < q_x ≤ 1, q_x crescente de 35 a 95 anos (senão, bloqueia).
3. mt > sb entre 20 e 80 anos (senão, bloqueia: provável troca de aba).
4. e_x publicado vs e_x calculado do q_x até 90 anos (diferença acima de 0,06 ano bloqueia).
5. q_x vs 1 − l(x+1)/l(x) entre 20 e 80 anos (diferença acima de 1% gera aviso com diagnóstico, não bloqueia).

## Qual versão usar

Para preço novo, a versão vigente na data da cotação (hoje, 2026). Para estudo de experiência, registre
no relatório qual versão serviu de base do A/E: trocar a base muda o nível do A/E, não a ordenação entre
grupos.
