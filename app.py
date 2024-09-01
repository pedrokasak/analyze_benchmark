import os
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.express as px
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.grid import grid

def build_sidebar():
    st.image(os.path.join('images', 'logo-250-100-transparente.png'))
    ticker_list = pd.read_csv('tickers_ibra.csv', index_col=0)
    tickers = st.multiselect(label='Selecione as empresas', options=ticker_list, placeholder='Selecione as empresas')
    tickers = [t+'.SA' for t in tickers]
    start_date = st.date_input(label='De', format='DD/MM/YYYY', value=datetime(2024,1,2), key='start_date')
    end_date = st.date_input(label='Até', format='DD/MM/YYYY', value="today", key='end_date')

    if tickers:
        prices = yf.download(tickers, start_date, end_date)['Adj Close']
        if len(tickers) == 1:
            prices = prices.to_frame()
            prices.columns = [tickers[0].rstrip('.SA')]

        prices.columns = prices.columns.str.rstrip('.SA')
        prices['IBOV'] = yf.download('^BVSP', start_date, end_date)['Adj Close']
        return tickers, prices
    return None,  None

def build_main(tickers, prices):
    weight = np.ones(len(tickers)) / len(tickers)
    prices['PORTFOLIO'] = prices.drop('IBOV', axis=1) @ weight
    normalize_prices = 100 * prices / prices.iloc[0]
    returns_prices  = prices.pct_change()[1:]
    volatility = returns_prices.std()* np.sqrt(252)
    total_returns = (normalize_prices.iloc[-1] - 100) / 100

    my_grid = grid(5,5,5,5,5,5, vertical_align='top')
    for ticker in prices.columns:
        cards = my_grid.container(border=True)
        cards.subheader(ticker, divider='red')
        colA, colB, colC = cards.columns(3)
        if ticker == 'PORTFOLIO':
            colA.image(os.path.join('images', 'pie-chart-dollar-svgrepo-com.svg'))
        elif ticker == 'IBOV':
            colA.image('images/pie-chart-svgrepo-com.svg')
        else:
            colA.image(f'https://raw.githubusercontent.com/thefintz/icones-b3/main/icones/{ticker}.png', width=60)
        colB.metric(label='Retorno', value=f'{total_returns[ticker]:.0%}', )
        colC.metric(label='Volatilidade', value=f'{volatility[ticker]:.0%}')
        style_metric_cards(background_color='rgba(255,255,255,0')

    col1, col2 = st.columns(2, gap='large')
    with col1:
        st.subheader(f'Desenpenho Relativo')
        st.line_chart(normalize_prices, height=350)

    with col2:
        st.subheader(f'Risco - Retorno')
        figure = px.scatter(
            x= volatility,
            y= total_returns,
            text=volatility.index,
            color= total_returns/volatility,
            color_continuous_scale=px.colors.sequential.Bluered_r
        )
        figure.update_traces(
            textfont_color = 'white',
            marker = dict(size=45),
            textfont_size = 10
        )
        figure.layout.yaxis.title = 'Retorno Total'
        figure.layout.xaxis.title = 'Volatilidade (anualizada)'
        figure.layout.height = 600
        figure.layout.xaxis.tickformat = '.0%'
        figure.layout.yaxis.tickformat = '.0%'
        figure.layout.coloraxis.colorbar.title = 'Sharpe'
        st.plotly_chart(figure, use_container_width=True)

st.set_page_config(page_title='Stock Analytics', layout='wide')

with st.sidebar:
    tickers, prices = build_sidebar()

st.title('Stock Analytics')
if tickers:
    build_main(tickers, prices)