# import yfinance as yf
# import pandas as pd
# import seaborn as sns
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error
# import matplotlib.pyplot as plt
#
#
# # Função para obter dados históricos
# def obter_dados_historicos(ticker, periodo='1y'):
#     stock = yf.Ticker(ticker)
#     dados = stock.history(period=periodo)
#     return dados
#
# # Taxa de juros atual (substitua pelo valor real)
# taxa_juros_atual = 11.34  # Exemplo: 6%
#
# # Função para calcular o valor intrínseco usando a fórmula de Graham
# def valor_intrinseco(eps, taxa_crescimento, taxa_juros=taxa_juros_atual):
#     return eps * (8.5 + 2 * taxa_crescimento) * 4.4 / taxa_juros
#
# # Função para obter dados financeiros
# def obter_dados_financeiros(ticker):
#     stock = yf.Ticker(ticker)
#     info = stock.info
#     eps = info.get('earningsPerShare', 0)
#     taxa_crescimento = info.get('earningsGrowth', 0)  # Taxa de crescimento dos lucros
#     preco_atual = info.get('currentPrice', 0)
#     return eps, taxa_crescimento, preco_atual
# # Lista de ações
# acoes = ['AURE3.SA', 'CEBR3.SA', 'CMIN3.SA']
#
# # Coletar dados históricos para cada ação
# dados_historicos = {}
# for acao in acoes:
#     dados_historicos[acao] = obter_dados_historicos(acao)
#
# # Exibir dados históricos de uma ação como exemplo
# print(dados_historicos['AURE3.SA'].head())
#
#
# # Preparar dados para treinamento
# def preparar_dados(dados):
#     dados['Data'] = dados.index
#     dados['Data'] = (dados['Data'] - dados['Data'].min())  # Normalizar as datas
#     dados['Data'] = dados['Data'].dt.days
#     X = dados[['Data']].values
#     y = dados['Close'].values
#     return X, y
#
#
# # Treinamento do modelo para uma ação
# def treinar_modelo(dados):
#     X, y = preparar_dados(dados)
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#
#     modelo = LinearRegression()
#     modelo.fit(X_train, y_train)
#
#     y_pred = modelo.predict(X_test)
#     mse = mean_squared_error(y_test, y_pred)
#     print(f'MSE: {mse}')
#
#     return modelo
#
#
# # Treinar o modelo para uma ação e fazer previsões
# modelo = treinar_modelo(dados_historicos['AURE3.SA'])
#
#
# # Fazer previsões
# def prever_preco(modelo, dados):
#     X, _ = preparar_dados(dados)
#     preco_previsto = modelo.predict(X)
#     return preco_previsto
#
#
# # Previsões para uma ação
# preco_previsto = prever_preco(modelo, dados_historicos['AURE3.SA'])
#
#
# # Visualização da distribuição de preços
# def visualizar_distribuicao_precos(dados):
#     plt.figure(figsize=(10, 6))
#     sns.histplot(dados['Close'], bins=30, kde=True)
#     plt.xlabel('Preço de Fechamento')
#     plt.ylabel('Frequência')
#     plt.title('Distribuição dos Preços de Fechamento')
#     plt.show()
#
#
# # Visualizar a distribuição de preços para uma ação
# visualizar_distribuicao_precos(dados_historicos['AURE3.SA'])
#
# # Plotar resultados
# plt.figure(figsize=(10, 6))
# plt.plot(dados_historicos['AURE3.SA'].index, dados_historicos['AURE3.SA']['Close'], label='Preço Real')
# plt.plot(dados_historicos['AURE3.SA'].index, preco_previsto, label='Preço Previsto', linestyle='--')
# plt.xlabel('Data')
# plt.ylabel('Preço')
# plt.title('Previsão de Preço de Ação')
# plt.legend()
# plt.show()

import yfinance as yf
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


def obter_dados_historicos(ticker, periodo='1y'):
    stock = yf.Ticker(ticker)
    dados = stock.history(period=periodo)
    return dados

# Função para calcular o valor intrínseco usando a fórmula de Graham
def valor_intrinseco(lpa, vpa):
    if lpa <= 0 or vpa <= 0:
        print(f"Valores inválidos para LPA ({lpa}) ou VPA ({vpa}) para o cálculo do valor intrínseco.")
        return None
    try:
        return np.sqrt(22.5 * lpa * vpa)
    except Exception as e:
        print(f"Erro ao calcular o valor intrínseco: {e}")
        return None

# Função para obter dados financeiros
def obter_dados_financeiros(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    lpa = info.get('trailingEps', 0)
    vpa = info.get('bookValue', 0)
    preco_atual = info.get('currentPrice', 0)
    return lpa, vpa, preco_atual

# Lista de ações
acoes = ['AURE3.SA', 'CEBR3.SA', 'CMIN3.SA', 'GOAU4.SA', 'HAPV3.SA', 'MOVI3.SA', 'MRFG3.SA', 'RAIZ4.SA']

# Coletar dados históricos e financeiros para cada ação
dados_historicos = {}
dados_financeiros = {}
for acao in acoes:
    dados_historicos[acao] = obter_dados_historicos(acao)
    dados_financeiros[acao] = obter_dados_financeiros(acao)

# Preparar dados para treinamento
def preparar_dados(dados):
    dados['Data'] = dados.index
    dados['Data'] = (dados['Data'] - dados['Data'].min())  # Normalizar as datas
    dados['Data'] = dados['Data'].dt.days
    X = dados[['Data']].values
    y = dados['Close'].values
    return X, y

# Treinamento do modelo para uma ação
def treinar_modelo(dados):
    X, y = preparar_dados(dados)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    modelo = LinearRegression()
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'MSE: {mse}')

    return modelo

# Função para prever preços usando o modelo
def prever_preco(modelo, dados):
    X, _ = preparar_dados(dados)
    preco_previsto = modelo.predict(X)
    return preco_previsto

# Função para prever preço futuro
def prever_preco_futuro(modelo, dados, meses_futuro):
    ultima_data = dados.index[-1]
    dias_futuros = meses_futuro * 30
    data_futura = ultima_data + timedelta(days=dias_futuros)
    X_futuro = np.array([(data_futura - dados.index.min()).days]).reshape(-1, 1)
    preco_futuro = modelo.predict(X_futuro)
    return preco_futuro[0]

# Função para visualizar a distribuição de preços
def visualizar_distribuicao_precos(dados, acao):
    plt.figure(figsize=(10, 6))
    sns.histplot(dados['Close'], bins=30, kde=True)
    plt.xlabel('Preço de Fechamento')
    plt.ylabel('Frequência')
    plt.title(f'Distribuição dos Preços de Fechamento para {acao}')
    plt.show()

# Função para visualizar a previsão de preços
def visualizar_previsao_precos(dados, preco_previsto, preco_futuro, acao):
    plt.figure(figsize=(12, 6))
    plt.plot(dados.index, dados['Close'], label='Preço Real')
    plt.plot(dados.index, preco_previsto, label='Preço Previsto', linestyle='--')
    plt.axvline(dados.index[-1] + timedelta(days=180), color='r', linestyle='--', label='Estimativa 6 Meses')
    plt.scatter(dados.index[-1] + timedelta(days=180), preco_futuro, color='r', zorder=5, label=f'Preço Estimado em 6 Meses: R${preco_futuro:.2f}')
    plt.xlabel('Data')
    plt.ylabel('Preço')
    plt.title(f'Previsão de Preço de Ação para {acao}')
    plt.legend()
    plt.show()

# Gerar gráficos e resultados para cada ação
for acao in acoes:
    dados = dados_historicos[acao]
    modelo = treinar_modelo(dados)
    preco_previsto = prever_preco(modelo, dados)

    # Prever o preço futuro
    preco_futuro = prever_preco_futuro(modelo, dados, meses_futuro=6)

    visualizar_distribuicao_precos(dados, acao)
    visualizar_previsao_precos(dados, preco_previsto, preco_futuro, acao)

    # Calcular e mostrar valor intrínseco usando a fórmula de Graham
    lpa, vpa, preco_atual = dados_financeiros[acao]
    valor_estimado = valor_intrinseco(lpa, vpa)
    if valor_estimado is not None:
        print(f'Valor Intrínseco Estimado para {acao}: R${valor_estimado:.2f}')
        print(f'Preço Atual de {acao}: R${preco_atual:.2f}')
        print(f'Diferença: R${valor_estimado - preco_atual:.2f}')
    else:
        print(f'Não foi possível calcular o valor intrínseco para {acao}.')
    print(f'Preço Estimado para {acao} daqui a 6 meses: R${preco_futuro:.2f}')