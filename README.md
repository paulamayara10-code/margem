# First Pricing Intelligence

App Streamlit para comparar o **Valor Bruto faturado** com o valor equivalente da tabela de preços, considerando somente operações cuja finalidade começa por **VENDA**.

## Regra automática pelo cadastro de clientes

O app cruza o campo `Cliente` do relatório RFATEQP01 com o campo `Codigo` do cadastro MATR021.

- **Revendedor:** usa o preço da coluna correspondente à **UF** da venda.
- **Consumidor Final:** usa o preço da coluna **Consumidor Final**.
- Tipos sem regra definida, clientes não encontrados ou códigos ambíguos ficam em **Pendências** e não distorcem os indicadores.

O cadastro define a **coluna de preço**. O filtro `Tipo de preço` continua definindo a linha da tabela: Venda Direta, Distribuidor ou Representante.

Quando um código possui lojas com tipos diferentes, o app tenta resolver pela UF e pelo nome. Casos ainda ambíguos podem ser definidos em `bases/mapa_clientes_excecao.csv`.

## Estrutura para o Git

```text
app.py
requirements.txt
README.md
bases/
  rfateqp01.xlsx
  Tabela de Precos(4).xlsx
  matr021.xlsx
  mapa_gerentes.csv
  mapa_clientes_excecao.csv
```

Os três arquivos Excel podem ficar fixos no Git. Para atualizar o app, substitua a base mantendo o mesmo nome e faça um novo commit.

Como as bases contêm informações internas, use um repositório privado.

## Execução local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud

- Branch: `main`
- Main file path: `app.py`

## Exportação

O relatório Excel contém:

- análise detalhada;
- resumo por vendedor;
- resumo por gerente;
- resumo por produto;
- resumo por tipo de cliente e referência aplicada;
- pendências de produto, cliente ou coluna de preço.
