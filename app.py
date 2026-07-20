from __future__ import annotations

import difflib
import io
import os
import re
import unicodedata
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# -----------------------------------------------------------------------------
# Configuração visual
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="First Pricing Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

FIRST_BLUE = "#0B2E59"
FIRST_BLUE_2 = "#0F5EA8"
FIRST_LIGHT = "#EAF3FB"
POSITIVE = "#138A5B"
NEGATIVE = "#CF3E4B"
WARNING = "#D98E04"
GRAY = "#6B7280"
APP_DIR = Path(__file__).resolve().parent

st.markdown(
    f"""
    <style>
        .stApp {{ background: #F6F8FB; }}
        [data-testid="stSidebar"] {{ background: #FFFFFF; border-right: 1px solid #E5E7EB; }}
        .block-container {{ padding-top: 1.25rem; padding-bottom: 2rem; max-width: 1650px; }}
        h1, h2, h3 {{ color: {FIRST_BLUE}; letter-spacing: -0.02em; }}
        div[data-testid="stMetric"] {{
            background: #FFFFFF;
            border: 1px solid #E5E7EB;
            border-radius: 14px;
            padding: 14px 16px;
            box-shadow: 0 2px 10px rgba(11, 46, 89, 0.05);
        }}
        div[data-testid="stMetric"] label {{ color: #536174; font-weight: 600; }}
        .hero {{
            background: linear-gradient(120deg, {FIRST_BLUE}, {FIRST_BLUE_2});
            border-radius: 18px;
            padding: 22px 26px;
            color: white;
            margin-bottom: 18px;
            box-shadow: 0 8px 28px rgba(11, 46, 89, 0.18);
        }}
        .hero h1 {{ color: white; margin: 0; font-size: 2rem; }}
        .hero p {{ margin: 5px 0 0 0; opacity: 0.92; }}
        .soft-card {{
            background: #FFFFFF;
            border: 1px solid #E5E7EB;
            border-radius: 14px;
            padding: 16px;
        }}
        .pill {{
            display: inline-block;
            padding: 5px 10px;
            border-radius: 999px;
            background: {FIRST_LIGHT};
            color: {FIRST_BLUE};
            font-size: 0.84rem;
            font-weight: 700;
            margin-right: 6px;
        }}
        .small-note {{ color: #667085; font-size: 0.86rem; }}
        [data-testid="stDataFrame"] {{ border-radius: 12px; overflow: hidden; }}
        .stDownloadButton button {{ border-radius: 10px; font-weight: 650; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Utilidades
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class SourceFile:
    name: str
    content: bytes


def normalize_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", text.strip().upper())


def canonical_code(value: Any) -> str:
    return re.sub(r"[^A-Z0-9]", "", normalize_text(value))


def base_code(value: Any) -> str:
    """Remove somente sufixos de versão claros, preservando o código principal."""
    text = normalize_text(value).replace(" ", "")
    previous = None
    while text != previous:
        previous = text
        text = re.sub(r"_(?:RV|TC|AT|C|R|V)\d*$", "", text)
        text = re.sub(r"_\d{1,3}$", "", text)
    text = re.sub(r"(?:RV|TC|AT)\d*$", "", text)
    return re.sub(r"[^A-Z0-9]", "", text)


def clean_column_name(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).replace("\xa0", " ").strip())


def to_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    cleaned = (
        series.astype(str)
        .str.replace("R$", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
        .replace({"": np.nan, "NAN": np.nan, "NONE": np.nan})
    )
    # Se a série original já veio com ponto decimal, a limpeza acima pode removê-lo.
    direct = pd.to_numeric(series, errors="coerce")
    converted = pd.to_numeric(cleaned, errors="coerce")
    return direct.where(direct.notna(), converted)


def parse_excel_dates(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    result = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    numeric_mask = numeric.notna() & numeric.between(20000, 90000)
    if numeric_mask.any():
        result.loc[numeric_mask] = pd.to_datetime(
            numeric.loc[numeric_mask], unit="D", origin="1899-12-30", errors="coerce"
        )
    text_mask = ~numeric_mask
    if text_mask.any():
        result.loc[text_mask] = pd.to_datetime(series.loc[text_mask], dayfirst=True, errors="coerce")
    return result


def fmt_currency(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "R$ 0,00"
    text = f"{float(value):,.2f}"
    return "R$ " + text.replace(",", "X").replace(".", ",").replace("X", ".")


def fmt_percent(value: float | int | None, decimals: int = 1) -> str:
    if value is None or pd.isna(value):
        return "0,0%"
    return f"{float(value) * 100:.{decimals}f}%".replace(".", ",")


def first_existing(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    lookup = {normalize_text(c): c for c in columns}
    for candidate in candidates:
        found = lookup.get(normalize_text(candidate))
        if found:
            return found
    return None


def local_source(candidates: list[str]) -> SourceFile | None:
    search_roots = [Path.cwd(), APP_DIR, APP_DIR / "bases"]
    checked: set[Path] = set()
    for candidate in candidates:
        candidate_path = Path(candidate)
        paths = [candidate_path] if candidate_path.is_absolute() else [
            root / candidate_path for root in search_roots
        ]
        for path in paths:
            try:
                resolved = path.resolve()
            except OSError:
                resolved = path
            if resolved in checked:
                continue
            checked.add(resolved)
            if path.exists() and path.is_file():
                return SourceFile(path.name, path.read_bytes())
    return None


def uploaded_or_local(uploaded: Any, candidates: list[str]) -> SourceFile | None:
    if uploaded is not None:
        return SourceFile(uploaded.name, uploaded.getvalue())
    return local_source(candidates)


@st.cache_data(show_spinner=False)
def workbook_sheet_names(content: bytes) -> list[str]:
    with pd.ExcelFile(io.BytesIO(content), engine="openpyxl") as xls:
        return list(xls.sheet_names)


@st.cache_data(show_spinner=False)
def read_excel_sheet(content: bytes, sheet_name: str) -> pd.DataFrame:
    frame = pd.read_excel(
        io.BytesIO(content),
        sheet_name=sheet_name,
        dtype=object,
        engine="openpyxl",
    )
    frame.columns = [clean_column_name(c) for c in frame.columns]
    return frame


def locate_faturamento_sheet(source: SourceFile) -> tuple[str, pd.DataFrame]:
    names = workbook_sheet_names(source.content)
    preferred = [n for n in names if normalize_text(n) != "PARAMETROS"] + names
    required = {"FINALIDADE", "PRODUTO", "PRC UNITARIO", "QUANTIDADE"}
    checked: set[str] = set()
    for sheet in preferred:
        if sheet in checked:
            continue
        checked.add(sheet)
        frame = read_excel_sheet(source.content, sheet)
        normalized = {normalize_text(c).split(".")[0] for c in frame.columns}
        if required.issubset(normalized):
            return sheet, frame
    raise ValueError("Não encontrei uma aba com Finalidade, Produto, Quantidade e Prc Unitario.")


def locate_price_sheet(source: SourceFile) -> tuple[str, pd.DataFrame]:
    names = workbook_sheet_names(source.content)
    preferred = [n for n in names if normalize_text(n) == "TABELA_UF"] + names
    checked: set[str] = set()
    for sheet in preferred:
        if sheet in checked:
            continue
        checked.add(sheet)
        frame = read_excel_sheet(source.content, sheet)
        normalized = {normalize_text(c) for c in frame.columns}
        if {"PRODUTO", "TIPO_PRECO"}.issubset(normalized):
            return sheet, frame
    raise ValueError("Não encontrei a aba Tabela_UF com Produto e TIPO_PRECO.")


def prepare_sales(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df.columns = [clean_column_name(c) for c in df.columns]

    aliases = {
        "Finalidade": ["Finalidade"],
        "Segmento": ["Segmento"],
        "Pedido": ["Numero", "Pedido", "Nro Pedido"],
        "Data": ["DT Emissao", "Data Emissao", "Emissao"],
        "Cliente_Codigo": ["Cliente", "Cod Cliente"],
        "Cliente": ["Nome Cliente", "Razao Social"],
        "UF": ["Estado", "UF"],
        "Vendedor": ["Nome", "Nome Vendedor", "Vendedor"],
        "Gerente": ["Gerente", "Nome Espec.", "Nome Espec", "Especialista"],
        "Produto": ["Produto"],
        "Descricao_Produto": ["Descricao"],
        "Quantidade": ["Quantidade"],
        "Preco_Unitario_Informado": ["Prc Unitario", "Preco Unitario"],
        "Valor_Liquido": ["Vlr.Total", "Valor Total"],
        "Valor_Bruto": ["Valor Bruto"],
        "Condicao_Pagamento": ["Cond. Pagto", "Condicao Pagamento"],
        "Descricao_Condicao": ["Descricao.1"],
        "Nota_Fiscal": ["Nota Fiscal", "NF"],
    }

    out = pd.DataFrame(index=df.index)
    for target, options in aliases.items():
        column = first_existing(df.columns, options)
        out[target] = df[column] if column else ""

    out["Finalidade"] = out["Finalidade"].fillna("").astype(str).str.strip()
    out["Finalidade_Normalizada"] = out["Finalidade"].map(normalize_text)
    out = out[out["Finalidade_Normalizada"].str.startswith("VENDA")].copy()

    out["Data"] = parse_excel_dates(out["Data"])
    for column in ["Quantidade", "Preco_Unitario_Informado", "Valor_Liquido", "Valor_Bruto"]:
        out[column] = to_numeric(out[column])

    out["Quantidade"] = out["Quantidade"].fillna(0)
    out["Preco_Unitario_Informado"] = out["Preco_Unitario_Informado"].fillna(0)
    calculated_total = out["Quantidade"] * out["Preco_Unitario_Informado"]
    out["Valor_Liquido"] = out["Valor_Liquido"].where(
        out["Valor_Liquido"].fillna(0) != 0, calculated_total
    )
    out["Valor_Bruto"] = out["Valor_Bruto"].fillna(0)

    # Regra principal solicitada: o realizado é o Valor Bruto do item.
    # O fallback só é usado quando o relatório vier sem Valor Bruto.
    out["Valor_Realizado"] = out["Valor_Bruto"].where(
        out["Valor_Bruto"] > 0, out["Valor_Liquido"]
    )
    out["Fonte_Valor_Realizado"] = np.where(
        out["Valor_Bruto"] > 0, "VALOR BRUTO", "VLR.TOTAL (FALLBACK)"
    )
    out["Preco_Realizado_Bruto"] = np.where(
        out["Quantidade"] > 0, out["Valor_Realizado"] / out["Quantidade"], np.nan
    )

    for column in [
        "Segmento",
        "Pedido",
        "Cliente_Codigo",
        "Cliente",
        "UF",
        "Vendedor",
        "Gerente",
        "Produto",
        "Descricao_Produto",
        "Condicao_Pagamento",
        "Descricao_Condicao",
        "Nota_Fiscal",
    ]:
        out[column] = out[column].fillna("").astype(str).str.strip()

    out["Gerente"] = out["Gerente"].replace("", "NÃO INFORMADO")
    out["UF"] = out["UF"].map(normalize_text)
    out["Produto_Canonico"] = out["Produto"].map(canonical_code)
    out["Produto_Base"] = out["Produto"].map(base_code)
    out = out[(out["Produto_Canonico"] != "") & (out["Valor_Realizado"] > 0)].copy()
    out.reset_index(drop=True, inplace=True)
    return out


def prepare_price_table(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df.columns = [clean_column_name(c) for c in df.columns]

    required = ["Produto", "TIPO_PRECO"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"A tabela de preços não possui: {', '.join(missing)}")

    df = df[df["Produto"].notna()].copy()
    df["Produto"] = df["Produto"].astype(str).str.strip()
    df = df[df["Produto"] != ""].copy()
    df["TIPO_PRECO"] = df["TIPO_PRECO"].fillna("").astype(str).str.strip()
    df["Produto_Canonico"] = df["Produto"].map(canonical_code)
    df["Produto_Base"] = df["Produto"].map(base_code)

    date_column = first_existing(df.columns, ["ATUALIZADO EM", "Atualizado em", "Data Atualizacao"])
    if date_column:
        df["Atualizado_Em"] = parse_excel_dates(df[date_column])
    else:
        df["Atualizado_Em"] = pd.NaT

    price_columns = ["Consumidor Final", "4%"] + [
        "AC", "AL", "AP", "AM", "BA", "CE", "DF", "ES", "GO", "MA", "MT",
        "MS", "MG", "PA", "PB", "PR", "PE", "PI", "RJ", "RN", "RS", "RO",
        "RR", "SC", "SP", "SE", "TO",
    ]
    for column in price_columns:
        if column in df.columns:
            df[column] = to_numeric(df[column])

    for column in ["GRUPO", "LINHA", "CLASSIFICAÇÃO", "Descrição"]:
        if column not in df.columns:
            df[column] = ""
        df[column] = df[column].fillna("").astype(str).str.strip()

    return df


def prepare_manual_mapping(uploaded: Any) -> dict[str, str]:
    if uploaded is None:
        return {}
    content = uploaded.getvalue()
    try:
        if uploaded.name.lower().endswith(".csv"):
            mapping_df = pd.read_csv(io.BytesIO(content), sep=None, engine="python", dtype=str)
        else:
            mapping_df = pd.read_excel(io.BytesIO(content), dtype=str, engine="openpyxl")
    except Exception as exc:
        st.sidebar.error(f"Não foi possível ler o mapa: {exc}")
        return {}

    mapping_df.columns = [clean_column_name(c) for c in mapping_df.columns]
    source_col = first_existing(mapping_df.columns, ["Produto_Faturamento", "Produto Faturamento", "De"])
    target_col = first_existing(mapping_df.columns, ["Produto_Tabela", "Produto Tabela", "Para"])
    if not source_col or not target_col:
        st.sidebar.error("O mapa precisa ter Produto_Faturamento e Produto_Tabela.")
        return {}
    mapping_df = mapping_df[[source_col, target_col]].dropna()
    return {
        canonical_code(source): str(target).strip()
        for source, target in mapping_df.itertuples(index=False, name=None)
        if canonical_code(source) and str(target).strip()
    }


def prepare_manager_mapping(source: SourceFile | None) -> dict[str, str]:
    if source is None:
        return {}
    try:
        if source.name.lower().endswith(".csv"):
            mapping_df = pd.read_csv(
                io.BytesIO(source.content), sep=None, engine="python", dtype=str
            )
        else:
            mapping_df = pd.read_excel(
                io.BytesIO(source.content), dtype=str, engine="openpyxl"
            )
    except Exception:
        return {}

    mapping_df.columns = [clean_column_name(c) for c in mapping_df.columns]
    seller_col = first_existing(mapping_df.columns, ["Vendedor", "Nome Vendedor"])
    manager_col = first_existing(mapping_df.columns, ["Gerente", "Nome Gerente"])
    if not seller_col or not manager_col:
        return {}

    mapping_df = mapping_df[[seller_col, manager_col]].fillna("")
    result: dict[str, str] = {}
    for seller, manager in mapping_df.itertuples(index=False, name=None):
        seller_key = normalize_text(seller)
        manager_name = str(manager).strip()
        if seller_key and manager_name:
            result[seller_key] = manager_name
    return result


def apply_manager_mapping(sales: pd.DataFrame, manager_mapping: dict[str, str]) -> pd.DataFrame:
    result = sales.copy()
    result["Gerente_Origem"] = np.where(
        result["Gerente"].eq("NÃO INFORMADO"),
        "NÃO INFORMADO",
        "RELATÓRIO — NOME ESPEC.",
    )
    if not manager_mapping:
        return result

    mapped = result["Vendedor"].map(normalize_text).map(manager_mapping)
    mask = mapped.notna() & mapped.astype(str).str.strip().ne("")
    result.loc[mask, "Gerente"] = mapped.loc[mask].astype(str).str.strip()
    result.loc[mask, "Gerente_Origem"] = "MAPA FIXO DO GIT"
    return result


def select_latest(records: pd.DataFrame) -> pd.Series:
    if records.empty:
        raise ValueError("Nenhum registro disponível")
    ordered = records.sort_values("Atualizado_Em", ascending=False, na_position="last")
    return ordered.iloc[0]


def build_indexes(table: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    raw_index = {key: group.copy() for key, group in table.groupby("Produto_Canonico", dropna=False)}
    base_index = {key: group.copy() for key, group in table.groupby("Produto_Base", dropna=False)}
    return raw_index, base_index


def match_product(
    sold_product: str,
    raw_index: dict[str, pd.DataFrame],
    base_index: dict[str, pd.DataFrame],
    manual_mapping: dict[str, str],
) -> tuple[pd.Series | None, str, str]:
    canonical = canonical_code(sold_product)
    base = base_code(sold_product)

    manual_target = manual_mapping.get(canonical)
    if manual_target:
        target_key = canonical_code(manual_target)
        records = raw_index.get(target_key)
        if records is not None and not records.empty:
            return select_latest(records), "MAPEAMENTO MANUAL", ""
        return None, "MAPEAMENTO INVÁLIDO", manual_target

    records = raw_index.get(canonical)
    if records is not None and not records.empty:
        return select_latest(records), "EXATO", ""

    records = raw_index.get(base)
    if records is not None and not records.empty:
        return select_latest(records), "SUFIXO IGNORADO", ""

    records = base_index.get(base)
    if records is None or records.empty:
        return None, "NÃO ENCONTRADO", ""

    distinct = records.drop_duplicates("Produto_Canonico")
    exact_base = distinct[distinct["Produto_Canonico"] == base]
    if not exact_base.empty:
        return select_latest(exact_base), "SUFIXO IGNORADO", ""
    if len(distinct) == 1:
        return select_latest(distinct), "EQUIVALENTE", ""

    candidates = ", ".join(distinct["Produto"].astype(str).head(8).tolist())
    return None, "AMBÍGUO", candidates


def classify_variation(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "SEM REFERÊNCIA"
    if value >= 0.10:
        return "ACIMA DE 10%"
    if value >= 0:
        return "ATÉ 10% ACIMA"
    if value >= -0.05:
        return "ATÉ 5% ABAIXO"
    if value >= -0.10:
        return "5% A 10% ABAIXO"
    return "MAIS DE 10% ABAIXO"


def compare_prices(
    sales: pd.DataFrame,
    table: pd.DataFrame,
    price_type: str,
    reference_mode: str,
    manual_mapping: dict[str, str],
) -> pd.DataFrame:
    selected_table = table[table["TIPO_PRECO"] == price_type].copy()
    raw_index, base_index = build_indexes(selected_table)

    rows: list[dict[str, Any]] = []
    for record in sales.to_dict("records"):
        matched, method, candidates = match_product(
            record["Produto"], raw_index, base_index, manual_mapping
        )
        result = dict(record)
        result["Metodo_Cruzamento"] = method
        result["Candidatos"] = candidates
        result["Tipo_Preco"] = price_type
        result["Produto_Tabela"] = ""
        result["Grupo"] = ""
        result["Linha"] = ""
        result["Classificacao"] = ""
        result["Descricao_Tabela"] = ""
        result["Atualizado_Em"] = pd.NaT
        result["Coluna_Referencia"] = ""
        result["Preco_Tabela"] = np.nan

        if matched is not None:
            result["Produto_Tabela"] = str(matched.get("Produto", ""))
            result["Grupo"] = str(matched.get("GRUPO", ""))
            result["Linha"] = str(matched.get("LINHA", ""))
            result["Classificacao"] = str(matched.get("CLASSIFICAÇÃO", ""))
            result["Descricao_Tabela"] = str(matched.get("Descrição", ""))
            result["Atualizado_Em"] = matched.get("Atualizado_Em", pd.NaT)

            if reference_mode == "Preço por UF":
                reference_column = record["UF"]
            elif reference_mode == "Consumidor Final":
                reference_column = "Consumidor Final"
            else:
                reference_column = "4%"
            result["Coluna_Referencia"] = reference_column
            price = matched.get(reference_column, np.nan)
            result["Preco_Tabela"] = pd.to_numeric(pd.Series([price]), errors="coerce").iloc[0]

        rows.append(result)

    result = pd.DataFrame(rows)
    result["Valor_Tabela"] = result["Preco_Tabela"] * result["Quantidade"]
    result["Diferenca_Total"] = result["Valor_Realizado"] - result["Valor_Tabela"]
    result["Impacto_Total"] = result["Diferenca_Total"]
    result["Variacao"] = np.where(
        result["Valor_Tabela"] > 0,
        result["Valor_Realizado"] / result["Valor_Tabela"] - 1,
        np.nan,
    )
    result["Indice_Preco"] = np.where(
        result["Valor_Tabela"] > 0,
        result["Valor_Realizado"] / result["Valor_Tabela"],
        np.nan,
    )
    result["Status"] = result["Variacao"].map(classify_variation)
    result["Mes"] = result["Data"].dt.to_period("M").dt.to_timestamp()
    return result


@st.cache_data(show_spinner=False)
def make_excel_export(analysis: pd.DataFrame) -> bytes:
    valid = analysis[
        analysis["Preco_Tabela"].notna() & (analysis["Valor_Tabela"] > 0)
    ].copy()
    by_seller = (
        valid.groupby(["Vendedor", "Gerente"], dropna=False)
        .agg(
            Faturamento_Bruto=("Valor_Realizado", "sum"),
            Valor_Tabela=("Valor_Tabela", "sum"),
            Impacto=("Impacto_Total", "sum"),
            Itens=("Produto", "size"),
        )
        .reset_index()
    )
    by_seller["Variacao_Ponderada"] = np.where(
        by_seller["Valor_Tabela"] != 0,
        by_seller["Faturamento_Bruto"] / by_seller["Valor_Tabela"] - 1,
        np.nan,
    )

    by_manager = (
        valid.groupby("Gerente", dropna=False)
        .agg(
            Faturamento_Bruto=("Valor_Realizado", "sum"),
            Valor_Tabela=("Valor_Tabela", "sum"),
            Impacto=("Impacto_Total", "sum"),
            Itens=("Produto", "size"),
        )
        .reset_index()
    )
    by_manager["Variacao_Ponderada"] = np.where(
        by_manager["Valor_Tabela"] != 0,
        by_manager["Faturamento_Bruto"] / by_manager["Valor_Tabela"] - 1,
        np.nan,
    )

    by_product = (
        valid.groupby(["Produto", "Produto_Tabela", "Descricao_Produto"], dropna=False)
        .agg(
            Quantidade=("Quantidade", "sum"),
            Faturamento_Bruto=("Valor_Realizado", "sum"),
            Valor_Tabela=("Valor_Tabela", "sum"),
            Impacto=("Impacto_Total", "sum"),
        )
        .reset_index()
    )
    by_product["Variacao_Ponderada"] = np.where(
        by_product["Valor_Tabela"] != 0,
        by_product["Faturamento_Bruto"] / by_product["Valor_Tabela"] - 1,
        np.nan,
    )

    pending = analysis[
        analysis["Preco_Tabela"].isna() | (analysis["Valor_Tabela"] <= 0)
    ].copy()

    export_columns = [
        "Data", "Nota_Fiscal", "Pedido", "Finalidade", "Segmento", "Cliente_Codigo",
        "Cliente", "UF", "Vendedor", "Gerente", "Produto", "Produto_Tabela",
        "Descricao_Produto", "Grupo", "Linha", "Classificacao", "Quantidade",
        "Preco_Unitario_Informado", "Preco_Realizado_Bruto", "Preco_Tabela",
        "Valor_Liquido", "Valor_Bruto", "Valor_Realizado", "Valor_Tabela",
        "Diferenca_Total", "Impacto_Total", "Variacao", "Status",
        "Fonte_Valor_Realizado", "Gerente_Origem", "Metodo_Cruzamento", "Candidatos", "Tipo_Preco",
        "Coluna_Referencia", "Atualizado_Em",
    ]
    export_data = analysis[[c for c in export_columns if c in analysis.columns]].copy()

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter", datetime_format="dd/mm/yyyy") as writer:
        export_data.to_excel(writer, sheet_name="Analise_Detalhada", index=False)
        by_seller.to_excel(writer, sheet_name="Resumo_Vendedor", index=False)
        by_manager.to_excel(writer, sheet_name="Resumo_Gerente", index=False)
        by_product.to_excel(writer, sheet_name="Resumo_Produto", index=False)
        pending.to_excel(writer, sheet_name="Pendencias", index=False)

        workbook = writer.book
        header_format = workbook.add_format(
            {"bold": True, "font_color": "#FFFFFF", "bg_color": FIRST_BLUE, "border": 0}
        )
        currency_format = workbook.add_format({"num_format": 'R$ #,##0.00;[Red]-R$ #,##0.00'})
        percent_format = workbook.add_format({"num_format": '0.0%;[Red]-0.0%'})
        date_format = workbook.add_format({"num_format": "dd/mm/yyyy"})

        sheets = {
            "Analise_Detalhada": export_data,
            "Resumo_Vendedor": by_seller,
            "Resumo_Gerente": by_manager,
            "Resumo_Produto": by_product,
            "Pendencias": pending,
        }
        for sheet_name, dataframe in sheets.items():
            worksheet = writer.sheets[sheet_name]
            worksheet.freeze_panes(1, 0)
            if len(dataframe.columns) > 0:
                worksheet.autofilter(
                    0, 0, max(len(dataframe), 1), len(dataframe.columns) - 1
                )
            worksheet.set_row(0, 24, header_format)
            for idx, column in enumerate(dataframe.columns):
                width = min(max(len(str(column)) + 2, 12), 34)
                sample = dataframe[column].astype("string").head(200)
                lengths = sample.str.len().replace([np.inf, -np.inf], np.nan).dropna()
                if not lengths.empty:
                    q90 = lengths.quantile(0.90)
                    if pd.notna(q90):
                        width = min(max(width, int(q90) + 2), 38)
                fmt = None
                if column in {
                    "Preco_Unitario_Informado", "Preco_Realizado_Bruto", "Preco_Tabela",
                    "Valor_Liquido", "Valor_Bruto", "Valor_Realizado", "Valor_Tabela",
                    "Diferenca_Total", "Impacto_Total", "Faturamento_Bruto", "Impacto",
                }:
                    fmt = currency_format
                elif column in {"Variacao", "Variacao_Ponderada"}:
                    fmt = percent_format
                elif column in {"Data", "Atualizado_Em"}:
                    fmt = date_format
                worksheet.set_column(idx, idx, width, fmt)

    return output.getvalue()


def suggestion_for_code(code: str, table_products: list[str]) -> str:
    canon_to_raw: dict[str, str] = {}
    for product in table_products:
        canon_to_raw.setdefault(canonical_code(product), product)
    matches = difflib.get_close_matches(canonical_code(code), list(canon_to_raw), n=3, cutoff=0.55)
    return " | ".join(canon_to_raw[m] for m in matches)


def chart_layout(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(
        title=title,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial", color="#344054"),
        margin=dict(l=20, r=20, t=55, b=20),
        legend_title_text="",
        hoverlabel=dict(bgcolor="white"),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#E8ECF2", zeroline=False)
    return fig


# -----------------------------------------------------------------------------
# Cabeçalho e entradas
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>First Pricing Intelligence</h1>
        <p>Comparativo do valor bruto faturado com o valor equivalente da tabela de preços</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Bases de análise")
    st.caption(
        "Sem upload, o app usa automaticamente os arquivos fixos salvos no Git. "
        "Um upload substitui a base apenas durante a sessão."
    )
    with st.expander("Substituir bases temporariamente", expanded=False):
        uploaded_sales = st.file_uploader("Faturamento", type=["xlsx", "xlsm", "xls"])
        uploaded_prices = st.file_uploader("Tabela de preços", type=["xlsx", "xlsm", "xls"])
        uploaded_mapping = st.file_uploader(
            "Mapa opcional de produtos",
            type=["xlsx", "csv"],
            help="Colunas esperadas: Produto_Faturamento e Produto_Tabela.",
        )

sales_source = uploaded_or_local(
    uploaded_sales,
    [
        "bases/rfateqp01.xlsx", "rfateqp01.xlsx", "RFATEQP01.xlsx",
        "bases/faturamento.xlsx", "faturamento.xlsx", "Faturamento.xlsx",
    ],
)
price_source = uploaded_or_local(
    uploaded_prices,
    [
        "bases/Tabela de Precos(4).xlsx", "Tabela de Precos(4).xlsx",
        "bases/Tabela de Precos.xlsx", "Tabela de Precos.xlsx",
        "bases/tabela_precos.xlsx", "tabela_precos.xlsx",
    ],
)

if sales_source is None or price_source is None:
    st.info(
        "Inclua no Git os arquivos `bases/rfateqp01.xlsx` e "
        "`bases/Tabela de Precos(4).xlsx`, ou envie as bases na lateral."
    )
    st.stop()

try:
    with st.spinner("Lendo e validando as bases..."):
        sales_sheet, sales_raw = locate_faturamento_sheet(sales_source)
        price_sheet, price_raw = locate_price_sheet(price_source)
        sales = prepare_sales(sales_raw)
        manager_map_source = local_source([
            "bases/mapa_gerentes.csv", "mapa_gerentes.csv",
            "bases/mapa_gerentes.xlsx", "mapa_gerentes.xlsx",
        ])
        manager_mapping = prepare_manager_mapping(manager_map_source)
        sales = apply_manager_mapping(sales, manager_mapping)
        price_table = prepare_price_table(price_raw)
        manual_mapping = prepare_manual_mapping(uploaded_mapping)
except Exception as exc:
    st.error(f"Não foi possível processar as bases: {exc}")
    st.stop()

if sales.empty:
    st.warning("Não encontrei operações cuja Finalidade começa por VENDA.")
    st.stop()

price_types = sorted([x for x in price_table["TIPO_PRECO"].dropna().unique() if str(x).strip()])
if not price_types:
    st.error("A coluna TIPO_PRECO está vazia na tabela de preços.")
    st.stop()

default_price_type = "Venda Direta" if "Venda Direta" in price_types else price_types[0]

with st.sidebar:
    st.divider()
    st.markdown("### Regra de comparação")
    price_type = st.selectbox(
        "Tipo de preço",
        price_types,
        index=price_types.index(default_price_type),
        help="Define qual linha da tabela será usada: Venda Direta, Distribuidor ou Representante.",
    )
    reference_mode = st.radio(
        "Preço de referência",
        ["Preço por UF", "Consumidor Final", "Alíquota 4%"],
        horizontal=False,
    )

    sale_operations = sorted(sales["Finalidade"].dropna().unique().tolist())
    selected_operations = st.multiselect(
        "Finalidades incluídas",
        sale_operations,
        default=sale_operations,
        help="O app já exclui operações que não começam por VENDA.",
    )

    valid_dates = sales["Data"].dropna()
    if not valid_dates.empty:
        min_date = valid_dates.min().date()
        max_date = valid_dates.max().date()
        selected_dates = st.date_input(
            "Período de emissão",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            format="DD/MM/YYYY",
        )
    else:
        selected_dates = None

    st.markdown("### Filtros principais")
    available_nfs = sorted([x for x in sales["Nota_Fiscal"].unique() if str(x).strip()])
    available_sellers = sorted([x for x in sales["Vendedor"].unique() if str(x).strip()])
    available_managers = sorted([x for x in sales["Gerente"].unique() if str(x).strip()])
    selected_nfs = st.multiselect("Nota fiscal", available_nfs, placeholder="Todas as NFs")
    selected_sellers_global = st.multiselect(
        "Vendedor", available_sellers, placeholder="Todos os vendedores"
    )
    selected_managers_global = st.multiselect(
        "Gerente", available_managers, placeholder="Todos os gerentes"
    )

    st.divider()
    sales_origin = "upload temporário" if uploaded_sales is not None else "arquivo fixo do Git"
    price_origin = "upload temporário" if uploaded_prices is not None else "arquivo fixo do Git"
    st.caption(f"Faturamento: {sales_source.name} • {sales_origin} • aba {sales_sheet}")
    st.caption(f"Tabela: {price_source.name} • {price_origin} • aba {price_sheet}")
    if manager_mapping:
        st.success(f"{len(manager_mapping)} vendedor(es) com gerente definido no mapa fixo.")
    else:
        st.caption("Gerente lido da coluna Nome Espec.; o mapa_gerentes.csv pode substituir essa regra.")
    if manual_mapping:
        st.success(f"{len(manual_mapping)} mapeamento(s) manual(is) de produto carregado(s).")

filtered_sales = sales[sales["Finalidade"].isin(selected_operations)].copy()
if selected_dates and isinstance(selected_dates, (tuple, list)) and len(selected_dates) == 2:
    start_date, end_date = selected_dates
    filtered_sales = filtered_sales[
        filtered_sales["Data"].dt.date.between(start_date, end_date, inclusive="both")
    ].copy()
if selected_nfs:
    filtered_sales = filtered_sales[filtered_sales["Nota_Fiscal"].isin(selected_nfs)].copy()
if selected_sellers_global:
    filtered_sales = filtered_sales[
        filtered_sales["Vendedor"].isin(selected_sellers_global)
    ].copy()
if selected_managers_global:
    filtered_sales = filtered_sales[
        filtered_sales["Gerente"].isin(selected_managers_global)
    ].copy()

if filtered_sales.empty:
    st.warning("Nenhuma venda encontrada para os filtros selecionados.")
    st.stop()

with st.spinner("Cruzando produtos e calculando as variações sobre o valor bruto..."):
    analysis = compare_prices(
        filtered_sales,
        price_table,
        price_type,
        reference_mode,
        manual_mapping,
    )

valid = analysis[
    analysis["Preco_Tabela"].notna() & (analysis["Valor_Tabela"] > 0)
].copy()
pending = analysis[
    analysis["Preco_Tabela"].isna() | (analysis["Valor_Tabela"] <= 0)
].copy()

revenue_total = analysis["Valor_Realizado"].sum()
matched_revenue = valid["Valor_Realizado"].sum()
reference_total = valid["Valor_Tabela"].sum()
impact_total = valid["Impacto_Total"].sum()
weighted_variation = matched_revenue / reference_total - 1 if reference_total else np.nan
coverage_rows = len(valid) / len(analysis) if len(analysis) else 0
coverage_revenue = matched_revenue / revenue_total if revenue_total else 0

st.markdown(
    f"""
    <span class="pill">{len(analysis):,} itens de venda</span>
    <span class="pill">{price_type}</span>
    <span class="pill">{reference_mode}</span>
    """.replace(",", "."),
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Abas
# -----------------------------------------------------------------------------
tab_summary, tab_detail, tab_pending, tab_method = st.tabs(
    ["Visão executiva", "Análise detalhada", "Pendências", "Metodologia"]
)

with tab_summary:
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Valor bruto das vendas", fmt_currency(revenue_total))
    k2.metric("Valor equivalente da tabela", fmt_currency(reference_total))
    k3.metric(
        "Ganho/perda vs tabela",
        fmt_currency(impact_total),
        delta=fmt_percent(weighted_variation),
        delta_color="normal",
    )
    k4.metric("Margem comercial vs tabela", fmt_percent(weighted_variation))
    k5.metric("Cobertura do cruzamento", fmt_percent(coverage_rows))

    st.caption(
        f"A variação ponderada considera o valor bruto apenas dos itens com preço de referência. "
        f"Cobertura por valor: {fmt_percent(coverage_revenue)}. "
        "A margem exibida é a diferença comercial do valor bruto frente à tabela, não margem bruta contábil."
    )

    if pending.empty:
        st.success("Todos os itens selecionados possuem preço de referência.")
    else:
        st.warning(
            f"{len(pending)} linha(s) não entraram no cálculo da variação por falta de preço ou cruzamento. "
            "Consulte a aba Pendências."
        )

    left, right = st.columns(2)

    with left:
        status_order = [
            "ACIMA DE 10%",
            "ATÉ 10% ACIMA",
            "ATÉ 5% ABAIXO",
            "5% A 10% ABAIXO",
            "MAIS DE 10% ABAIXO",
        ]
        status_colors = {
            "ACIMA DE 10%": POSITIVE,
            "ATÉ 10% ACIMA": "#56A47B",
            "ATÉ 5% ABAIXO": WARNING,
            "5% A 10% ABAIXO": "#E17A45",
            "MAIS DE 10% ABAIXO": NEGATIVE,
        }
        status_summary = (
            valid.groupby("Status", dropna=False)
            .agg(Itens=("Produto", "size"), Valor=("Valor_Realizado", "sum"))
            .reindex(status_order)
            .dropna(how="all")
            .reset_index()
        )
        if not status_summary.empty:
            fig = px.pie(
                status_summary,
                names="Status",
                values="Itens",
                hole=0.62,
                color="Status",
                color_discrete_map=status_colors,
            )
            fig.update_traces(textposition="inside", textinfo="percent+label")
            fig = chart_layout(fig, "Distribuição dos itens por variação")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with right:
        monthly = (
            valid.dropna(subset=["Mes"])
            .groupby("Mes", as_index=False)
            .agg(Vendido=("Valor_Realizado", "sum"), Tabela=("Valor_Tabela", "sum"))
        )
        if not monthly.empty:
            melted = monthly.melt("Mes", var_name="Série", value_name="Valor")
            fig = px.line(
                melted,
                x="Mes",
                y="Valor",
                color="Série",
                markers=True,
                color_discrete_map={"Vendido": FIRST_BLUE_2, "Tabela": GRAY},
            )
            fig.update_yaxes(tickprefix="R$ ", tickformat=".2s")
            fig = chart_layout(fig, "Valor bruto realizado x tabela por mês")
            st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)
    with left:
        seller_summary = (
            valid.groupby("Vendedor", as_index=False)
            .agg(
                Faturamento_Bruto=("Valor_Realizado", "sum"),
                Tabela=("Valor_Tabela", "sum"),
                Impacto=("Impacto_Total", "sum"),
            )
        )
        seller_summary["Variacao"] = np.where(
            seller_summary["Tabela"] > 0,
            seller_summary["Faturamento_Bruto"] / seller_summary["Tabela"] - 1,
            np.nan,
        )
        seller_summary = seller_summary.sort_values("Impacto").tail(15)
        if not seller_summary.empty:
            seller_summary["Cor"] = np.where(seller_summary["Impacto"] >= 0, "Positivo", "Negativo")
            fig = px.bar(
                seller_summary,
                x="Impacto",
                y="Vendedor",
                orientation="h",
                color="Cor",
                color_discrete_map={"Positivo": POSITIVE, "Negativo": NEGATIVE},
                custom_data=["Faturamento_Bruto", "Tabela", "Variacao"],
            )
            fig.update_traces(
                hovertemplate=(
                    "<b>%{y}</b><br>Impacto: R$ %{x:,.2f}<br>"
                    "Vendido: R$ %{customdata[0]:,.2f}<br>"
                    "Tabela: R$ %{customdata[1]:,.2f}<br>"
                    "Variação: %{customdata[2]:.1%}<extra></extra>"
                )
            )
            fig.update_layout(showlegend=False)
            fig.update_xaxes(tickprefix="R$ ", tickformat=".2s")
            fig = chart_layout(fig, "Impacto financeiro por vendedor")
            st.plotly_chart(fig, use_container_width=True)

    with right:
        negative_products = (
            valid.groupby(["Produto", "Descricao_Produto"], as_index=False)
            .agg(Impacto=("Impacto_Total", "sum"), Faturamento_Bruto=("Valor_Realizado", "sum"))
            .sort_values("Impacto")
            .head(15)
        )
        if not negative_products.empty:
            negative_products["Rótulo"] = negative_products.apply(
                lambda row: f"{row['Produto']} — {str(row['Descricao_Produto'])[:38]}", axis=1
            )
            fig = px.bar(
                negative_products.sort_values("Impacto", ascending=False),
                x="Impacto",
                y="Rótulo",
                orientation="h",
                color_discrete_sequence=[NEGATIVE],
            )
            fig.update_layout(showlegend=False)
            fig.update_xaxes(tickprefix="R$ ", tickformat=".2s")
            fig = chart_layout(fig, "Produtos com maior perda frente à tabela")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Principais pontos de atenção")
    attention = valid[valid["Variacao"] < -0.10].copy()
    if attention.empty:
        st.success("Nenhum item foi vendido mais de 10% abaixo da referência selecionada.")
    else:
        top_attention = (
            attention.groupby(["Gerente", "Vendedor", "Cliente", "Produto", "Descricao_Produto"], as_index=False)
            .agg(
                Faturamento_Bruto=("Valor_Realizado", "sum"),
                Valor_Tabela=("Valor_Tabela", "sum"),
                Impacto=("Impacto_Total", "sum"),
                Menor_Variacao=("Variacao", "min"),
            )
            .sort_values("Impacto")
            .head(12)
        )
        top_attention_display = top_attention.copy()
        top_attention_display["Menor_Variacao"] = top_attention_display["Menor_Variacao"] * 100
        st.dataframe(
            top_attention_display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Faturamento_Bruto": st.column_config.NumberColumn("Faturamento bruto", format="R$ %.2f"),
                "Valor_Tabela": st.column_config.NumberColumn("Tabela", format="R$ %.2f"),
                "Impacto": st.column_config.NumberColumn(format="R$ %.2f"),
                "Menor_Variacao": st.column_config.NumberColumn("Menor variação", format="%.1f%%"),
            },
        )

with tab_detail:
    st.markdown("### Filtros da análise detalhada")
    f1, f2, f3, f4, f5 = st.columns(5)
    sellers = sorted([x for x in analysis["Vendedor"].dropna().unique() if str(x).strip()])
    managers = sorted([x for x in analysis["Gerente"].dropna().unique() if str(x).strip()])
    segments = sorted([x for x in analysis["Segmento"].dropna().unique() if str(x).strip()])
    statuses = sorted([x for x in analysis["Status"].dropna().unique() if str(x).strip()])
    methods = sorted([x for x in analysis["Metodo_Cruzamento"].dropna().unique() if str(x).strip()])
    selected_sellers = f1.multiselect("Vendedor", sellers)
    selected_managers = f2.multiselect("Gerente", managers)
    selected_segments = f3.multiselect("Segmento", segments)
    selected_statuses = f4.multiselect("Status", statuses)
    selected_methods = f5.multiselect("Cruzamento", methods)
    search = st.text_input("Buscar cliente, vendedor, gerente, produto, pedido ou nota fiscal")

    detail = analysis.copy()
    if selected_sellers:
        detail = detail[detail["Vendedor"].isin(selected_sellers)]
    if selected_managers:
        detail = detail[detail["Gerente"].isin(selected_managers)]
    if selected_segments:
        detail = detail[detail["Segmento"].isin(selected_segments)]
    if selected_statuses:
        detail = detail[detail["Status"].isin(selected_statuses)]
    if selected_methods:
        detail = detail[detail["Metodo_Cruzamento"].isin(selected_methods)]
    if search.strip():
        needle = normalize_text(search)
        haystack = (
            detail[["Cliente", "Vendedor", "Gerente", "Produto", "Descricao_Produto", "Pedido", "Nota_Fiscal"]]
            .fillna("")
            .astype(str)
            .agg(" ".join, axis=1)
            .map(normalize_text)
        )
        detail = detail[haystack.str.contains(re.escape(needle), na=False)]

    display_columns = [
        "Data", "Nota_Fiscal", "Pedido", "Finalidade", "Cliente", "UF", "Vendedor", "Gerente",
        "Produto", "Produto_Tabela", "Descricao_Produto", "Quantidade", "Valor_Bruto",
        "Valor_Tabela", "Preco_Tabela", "Variacao", "Diferenca_Total",
        "Impacto_Total", "Status",
        "Metodo_Cruzamento",
    ]
    detail_display = detail[display_columns].sort_values(
        ["Data", "Impacto_Total"], ascending=[False, True]
    ).copy()
    detail_display["Variacao"] = detail_display["Variacao"] * 100
    st.dataframe(
        detail_display,
        use_container_width=True,
        hide_index=True,
        height=620,
        column_config={
            "Data": st.column_config.DateColumn("Emissão", format="DD/MM/YYYY"),
            "Nota_Fiscal": "NF",
            "Produto_Tabela": "Produto tabela",
            "Descricao_Produto": "Descrição",
            "Valor_Bruto": st.column_config.NumberColumn("Valor bruto", format="R$ %.2f"),
            "Valor_Tabela": st.column_config.NumberColumn("Valor tabela", format="R$ %.2f"),
            "Preco_Tabela": st.column_config.NumberColumn("Preço tabela", format="R$ %.2f"),
            "Variacao": st.column_config.NumberColumn("Variação", format="%.1f%%"),
            "Diferenca_Total": st.column_config.NumberColumn("Diferença total", format="R$ %.2f"),
            "Impacto_Total": st.column_config.NumberColumn("Impacto total", format="R$ %.2f"),
            "Metodo_Cruzamento": "Cruzamento",
        },
    )

    d1, d2 = st.columns([1, 1])
    csv_data = detail[display_columns].to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig")
    d1.download_button(
        "Baixar seleção em CSV",
        csv_data,
        file_name="comparativo_precos_selecao.csv",
        mime="text/csv",
        use_container_width=True,
    )
    excel_data = make_excel_export(analysis)
    d2.download_button(
        "Baixar relatório completo em Excel",
        excel_data,
        file_name="first_pricing_intelligence.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

with tab_pending:
    st.markdown("### Itens sem preço de referência")
    st.caption(
        "Esses itens ficam fora dos indicadores de variação. O app não força um cruzamento quando há risco de usar o produto errado."
    )

    if pending.empty:
        st.success("Não há pendências para os filtros atuais.")
    else:
        table_products = sorted(
            price_table.loc[price_table["TIPO_PRECO"] == price_type, "Produto"].astype(str).unique().tolist()
        )
        pending_summary = (
            pending.groupby(["Produto", "Descricao_Produto", "Metodo_Cruzamento", "Candidatos"], dropna=False)
            .agg(
                Linhas=("Produto", "size"),
                Quantidade=("Quantidade", "sum"),
                Faturamento_Bruto=("Valor_Realizado", "sum"),
            )
            .reset_index()
            .sort_values("Faturamento_Bruto", ascending=False)
        )
        pending_summary["Sugestao"] = pending_summary["Produto"].map(
            lambda code: suggestion_for_code(code, table_products)
        )
        st.dataframe(
            pending_summary,
            use_container_width=True,
            hide_index=True,
            height=520,
            column_config={
                "Descricao_Produto": "Descrição",
                "Metodo_Cruzamento": "Situação",
                "Faturamento_Bruto": st.column_config.NumberColumn("Faturamento bruto", format="R$ %.2f"),
                "Sugestao": "Possíveis códigos",
            },
        )

        mapping_template = pending_summary[["Produto"]].drop_duplicates().rename(
            columns={"Produto": "Produto_Faturamento"}
        )
        mapping_template["Produto_Tabela"] = ""
        mapping_template["Observacao"] = ""
        st.download_button(
            "Baixar modelo para correção dos cruzamentos",
            mapping_template.to_csv(index=False, sep=";").encode("utf-8-sig"),
            file_name="mapa_produtos.csv",
            mime="text/csv",
        )

with tab_method:
    st.markdown("### Regras aplicadas")
    st.markdown(
        """
        **1. Operações analisadas**  
        São consideradas apenas as linhas cuja coluna **Finalidade começa por “VENDA”**. Assim, locação, cobrança, remessa, devolução e serviço não entram no cálculo. Remessas de venda para entrega futura também não entram, pois começam por “REM”.

        **2. Valor realizado**  
        O comparativo utiliza a coluna **Valor Bruto** do item faturado. Quando essa coluna estiver vazia ou zerada, o app usa **Vlr.Total** apenas como fallback e identifica essa origem no relatório exportado.

        **3. Valor de referência**  
        A tabela é filtrada pelo **TIPO_PRECO** escolhido. O preço unitário de referência é obtido pela UF do cliente, por Consumidor Final ou pela coluna 4%. Em seguida, o app calcula **preço tabela × quantidade** para chegar ao valor total comparável.

        **4. Cruzamento dos produtos**  
        A ordem é: código exato; código com pontuação ignorada; código-base com sufixos como `_RV`, `_TC`, `_AT`, `_01` e similares ignorados; e equivalente único. Se houver mais de um candidato, o item fica como ambíguo e não entra no cálculo.

        **5. Indicadores**  
        - **Variação:** valor bruto realizado ÷ valor equivalente da tabela − 1.  
        - **Ganho/perda total:** valor bruto realizado − valor equivalente da tabela.  
        - **Margem comercial vs tabela:** valor bruto cruzado ÷ valor de tabela cruzado − 1.  
        - **Gerente:** o app prioriza o arquivo fixo **bases/mapa_gerentes.csv**; quando não houver mapeamento, utiliza a coluna **Nome Espec.** do relatório. Registros vazios aparecem como **NÃO INFORMADO**.

        A margem apresentada não é a margem bruta contábil, porque as bases fornecidas não possuem custo do produto.
        """
    )

st.markdown(
    "<div class='small-note' style='text-align:center;margin-top:22px;'>Desenvolvido para análise comercial e controladoria • First Medical</div>",
    unsafe_allow_html=True,
)
