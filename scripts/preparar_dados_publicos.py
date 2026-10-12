"""Prepara as bases públicas usadas no projeto (roda na máquina do usuário).

1. SUSEP SES: filtra Ses_seguros.csv nos ramos de pessoas e copia os dicionários de códigos.
2. SOA 2016 Group Term Life: extrai os dados de origem (cache) das tabelas dinâmicas .xlsx. Os arquivos .xlsb
   (binários) são convertidos para .xlsx pelo próprio Excel, por automação (Windows + Excel + pywin32).

Uso (tudo numa linha só):
  python scripts/preparar_dados_publicos.py --ses "C:/.../BaseCompleta" --soa "C:/.../group-life-10-13-pivot-tables" --saida dados_publicos
  python scripts/preparar_dados_publicos.py --agregar-soa --saida dados_publicos   (reduz o arquivo grande da SOA)

Saída: arquivos .csv.gz pequenos na pasta --saida e um resumo impresso na tela.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import re
import shutil
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
RAMOS_PESSOAS = {"0993", "0977", "0982"}       # vida, prestamista, acidentes pessoais coletivo
DICIONARIOS_SES = ["Ses_campos.csv", "Ses_ramos.csv", "Ses_cias.csv", "ses_gruposramos.csv"]


def _separador(caminho: Path, encoding: str) -> str:
    with open(caminho, encoding=encoding, errors="replace") as f:
        linha = f.readline()
    return ";" if linha.count(";") >= linha.count(",") else ","


def preparar_ses(pasta: Path, saida: Path) -> None:
    import pandas as pd
    arq = next((p for p in pasta.iterdir() if p.name.lower() == "ses_seguros.csv"), None)
    if arq is None:
        raise FileNotFoundError(f"Ses_seguros.csv não encontrado em {pasta}")
    enc = "latin-1"
    sep = _separador(arq, enc)
    cab = pd.read_csv(arq, sep=sep, encoding=enc, nrows=3, dtype=str)
    print("SES colunas:", list(cab.columns))
    cols_ramo = [c for c in cab.columns if "ramo" in c.lower()]
    if not cols_ramo:
        raise KeyError("Nenhuma coluna com 'ramo' no nome; envie a lista de colunas acima.")
    col = cols_ramo[0]
    print("Coluna de ramo usada:", col)
    partes, total = [], 0
    print("Lendo Ses_seguros.csv em blocos (pode levar alguns minutos)...", flush=True)
    for bloco in pd.read_csv(arq, sep=sep, encoding=enc, dtype=str, chunksize=500_000):
        total += len(bloco)
        print(f"  {total:,} linhas lidas", flush=True)
        ramo = bloco[col].astype(str).str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(4)
        partes.append(bloco[ramo.isin(RAMOS_PESSOAS)])
    df = pd.concat(partes, ignore_index=True)
    destino = saida / "ses_seguros_pessoas.csv.gz"
    df.to_csv(destino, index=False, sep=";", encoding="utf-8", compression="gzip")
    print(f"SES: {total:,} linhas lidas, {len(df):,} nos ramos {sorted(RAMOS_PESSOAS)} -> {destino.name}")
    for nome in DICIONARIOS_SES:
        p = next((q for q in pasta.iterdir() if q.name.lower() == nome.lower()), None)
        if p is not None:
            shutil.copy(p, saida / p.name)
            print("  copiado:", p.name)
        else:
            print("  não encontrado (ok se não existir):", nome)


def extrair_cache_pivot(xlsx: Path, saida: Path) -> list[Path]:
    """Extrai cada cache de tabela dinâmica do .xlsx para um .csv.gz (uma linha por registro de origem)."""
    gerados = []
    with zipfile.ZipFile(xlsx) as z:
        nomes = set(z.namelist())
        defs = sorted(n for n in nomes if re.fullmatch(r"xl/pivotCache/pivotCacheDefinition\d+\.xml", n))
        for d in defs:
            k = re.search(r"(\d+)\.xml$", d).group(1)
            rec = f"xl/pivotCache/pivotCacheRecords{k}.xml"
            if rec not in nomes:
                continue
            raiz = ET.fromstring(z.read(d))
            campos, compartilhados = [], []
            for cf in raiz.iter(NS + "cacheField"):
                if cf.get("databaseField", "1") == "0":      # campos calculados/agrupados não vão para os registros
                    continue
                campos.append(cf.get("name"))
                si = cf.find(NS + "sharedItems")
                compartilhados.append([("" if it.tag == NS + "m" else it.get("v")) for it in (si if si is not None else [])])
            base = re.sub(r"[^A-Za-z0-9]+", "_", xlsx.stem).strip("_")
            destino = saida / f"soa_{base}_cache{k}.csv.gz"
            n = 0
            with z.open(rec) as f, gzip.open(destino, "wt", newline="", encoding="utf-8") as g:
                w = csv.writer(g)
                w.writerow(campos)
                contexto = ET.iterparse(f, events=("start", "end"))
                _, topo = next(contexto)
                for evento, el in contexto:
                    if evento == "end" and el.tag == NS + "r":
                        linha = []
                        for i, c in enumerate(el):
                            tag = c.tag[len(NS):]
                            if tag == "x":
                                linha.append(compartilhados[i][int(c.get("v"))])
                            elif tag == "m":
                                linha.append("")
                            else:                               # n (número), s (texto), b, d, e
                                linha.append(c.get("v"))
                        w.writerow(linha)
                        n += 1
                        topo.clear()
            print(f"SOA {xlsx.name} cache {k}: {n:,} registros, {len(campos)} campos -> {destino.name}")
            print("   campos:", campos)
            gerados.append(destino)
    if not gerados:
        print(f"SOA {xlsx.name}: nenhum cache de tabela dinâmica encontrado")
    return gerados


def converter_xlsb_com_excel(xlsb: Path, pasta_destino: Path) -> Path | None:
    """Abre o .xlsb no Excel (invisível) e salva como .xlsx, preservando as tabelas dinâmicas e o cache.
    Não existe leitor Python maduro para o cache de tabela dinâmica do formato binário; o Excel é o
    conversor confiável. Requer Windows, Excel instalado e `pip install pywin32`."""
    try:
        import win32com.client as win32
    except ImportError:
        print("  pywin32 não instalado: rode `pip install pywin32` e execute o script de novo.")
        return None
    pasta_destino.mkdir(parents=True, exist_ok=True)
    destino = (pasta_destino / (xlsb.stem + ".xlsx")).resolve()
    if destino.exists():
        print(f"  já convertido anteriormente: {destino.name}")
        return destino
    excel = win32.DispatchEx("Excel.Application")
    excel.Visible = False
    excel.DisplayAlerts = False
    try:
        print(f"  convertendo {xlsb.name} com o Excel (pode levar alguns minutos)...", flush=True)
        wb = excel.Workbooks.Open(str(xlsb.resolve()), ReadOnly=True)
        for i in range(1, wb.PivotCaches().Count + 1):      # garante que os dados de origem sejam salvos
            try:
                wb.PivotCaches(i).SaveData = True
            except Exception:
                pass
        wb.SaveAs(str(destino), FileFormat=51)              # 51 = xlOpenXMLWorkbook (.xlsx)
        wb.Close(SaveChanges=False)
    finally:
        excel.Quit()
    print(f"  convertido: {destino.name}")
    return destino


def preparar_soa(pasta: Path, saida: Path) -> None:
    alvo = lambda p: "basic life death" in p.name.lower()
    xlsx = [p for p in pasta.iterdir() if p.suffix.lower() == ".xlsx" and alvo(p)]
    xlsb = [p for p in pasta.iterdir() if p.suffix.lower() == ".xlsb" and alvo(p)]
    for p in xlsb:
        if any(x.stem == p.stem for x in xlsx):              # já existe .xlsx com o mesmo nome
            continue
        convertido = converter_xlsb_com_excel(p, saida / "_convertidos")
        if convertido is not None:
            xlsx.append(convertido)
        else:
            print(f"  {p.name} não foi convertido. Alternativa manual: abrir no Excel e salvar como .xlsx.")
    if not xlsx:
        raise FileNotFoundError("Nenhum arquivo de 'Basic Life Death...' disponível em .xlsx")
    for p in xlsx:
        extrair_cache_pivot(p, saida)


# variáveis usadas no projeto (mortalidade do seguro básico); o resto do arquivo (invalidez, waiver, LTD,
# região) não entra no modelo e só aumentaria o tamanho
CHAVES_SOA = ["Type_of_Coverage", "exposure_coverage_type", "industry", "industry_cat", "face_amount_band",
              "salary", "group_size", "sex", "central_age"]
SOMAS_SOA = ["death_policies_exposed", "death_amount_exposed", "death_count", "death_claim_amount",
             "expected_death_by_policy", "expected_death_by_amount"]


def agregar_soa(saida: Path) -> None:
    """Agrega o cache grande da SOA (milhões de linhas) somando exposição, óbitos e esperados de morte
    pelas variáveis do modelo. Gera um arquivo pequeno e mostra os níveis de cada variável."""
    import pandas as pd
    grande = next((p for p in saida.glob("soa_*Basic_Life_Death_and_Dis_AE_Ratios_Final_cache*.csv.gz")), None)
    if grande is None:
        raise FileNotFoundError("Arquivo grande da SOA não encontrado; rode antes com --soa")
    print(f"Agregando {grande.name} (pode levar alguns minutos)...", flush=True)
    partes, total = [], 0
    for bloco in pd.read_csv(grande, dtype=str, chunksize=1_000_000, keep_default_na=False):
        total += len(bloco)
        chaves = [c for c in CHAVES_SOA if c in bloco.columns]
        for c in SOMAS_SOA:
            bloco[c] = pd.to_numeric(bloco[c], errors="coerce").fillna(0.0)
        partes.append(bloco.groupby(chaves, as_index=False)[SOMAS_SOA].sum())
        print(f"  {total:,} linhas lidas", flush=True)
    df = pd.concat(partes, ignore_index=True).groupby(chaves, as_index=False)[SOMAS_SOA].sum()
    df = df[df["death_policies_exposed"] > 0]
    destino = saida / "soa_basic_life_morte_agregado.csv.gz"
    df.to_csv(destino, index=False, compression="gzip")
    print(f"SOA agregado: {total:,} linhas -> {len(df):,} células -> {destino.name}")
    print("Totais:", {c: round(float(df[c].sum()), 1) for c in ["death_policies_exposed", "death_count"]})
    for c in chaves:
        niveis = df.groupby(c)["death_policies_exposed"].sum().sort_values(ascending=False)
        amostra = ", ".join(f"{k} ({v / niveis.sum():.1%})" for k, v in niveis.head(30).items())
        print(f"  {c}: {len(niveis)} níveis -> {amostra}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ses", type=Path, help="pasta BaseCompleta do SES")
    ap.add_argument("--soa", type=Path, help="pasta group-life-10-13-pivot-tables")
    ap.add_argument("--saida", type=Path, default=Path("dados_publicos"))
    ap.add_argument("--agregar-soa", action="store_true", help="agrega o arquivo grande da SOA já extraído")
    a = ap.parse_args()
    if not a.ses and not a.soa and not a.agregar_soa:
        ap.error("informe --ses, --soa ou --agregar-soa (nada a fazer)")
    for nome, pasta in (("--ses", a.ses), ("--soa", a.soa)):
        if pasta is not None and not pasta.is_dir():
            ap.error(f"{nome}: pasta não encontrada: {pasta}")
    a.saida.mkdir(parents=True, exist_ok=True)
    print("Pasta de saída:", a.saida.resolve())
    if a.ses:
        preparar_ses(a.ses, a.saida)
    if a.soa:
        preparar_soa(a.soa, a.saida)
    if a.agregar_soa:
        agregar_soa(a.saida)
    print("\nArquivos gerados:")
    for p in sorted(x for x in a.saida.iterdir() if x.is_file()):
        print(f"  {p.name:70s} {p.stat().st_size / 1e6:8.2f} MB")


if __name__ == "__main__":
    main()
