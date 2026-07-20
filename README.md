# First Pricing Intelligence

App Streamlit para comparar operações de venda do relatório de faturamento com a tabela de preços por UF.

## Arquivos

- `app_comparativo_precos.py`: aplicação principal.
- `requirements_comparativo_precos.txt`: dependências.

## Como publicar no Streamlit

1. Envie os dois arquivos acima para um repositório GitHub.
2. Renomeie `requirements_comparativo_precos.txt` para `requirements.txt`.
3. No Streamlit Community Cloud, selecione `app_comparativo_precos.py` como arquivo principal.
4. As bases podem ser enviadas pela tela do app. Opcionalmente, podem ficar no repositório com um destes nomes:
   - `rfateqp01.xlsx`
   - `Tabela de Precos.xlsx`

## Regras da versão inicial

- Inclui somente finalidades que começam por `VENDA`.
- Exclui remessas, locações, serviços e cobranças.
- Utiliza Produto, Quantidade, Prc Unitario e Vlr.Total do primeiro conjunto de colunas do relatório.
- Permite escolher Venda Direta, Distribuidor ou Representante.
- Permite comparar por UF, Consumidor Final ou coluna 4%.
- Ignora pontuação e sufixos claros de versão, sem forçar cruzamentos ambíguos.
- Exporta relatório completo em Excel e seleção em CSV.

## Mapa opcional de produtos

O app aceita um arquivo CSV ou Excel com as colunas:

- `Produto_Faturamento`
- `Produto_Tabela`

Use o modelo disponível na aba Pendências do próprio app.
