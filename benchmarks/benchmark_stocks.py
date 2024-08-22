import json
import os
from datetime import datetime
from dotenv import load_dotenv, dotenv_values

import yfinance as yf

from crewai import Agent, Task, Crew, Process

from prophet import Prophet
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchResults
import pandas as pd
import streamlit as st

load_dotenv()


def fetch_stock_price(ticket):
    stock = yf.download(ticket, start="2020-08-08", end="2024-08-08")
    return stock


# Função para prever o preço da ação usando Prophet
def predict_stock_price(stock_data):
    df = pd.DataFrame({"ds": stock_data.index, "y": stock_data["Close"]})

    model = Prophet()
    model.fit(df)

    future = model.make_future_dataframe(periods=365)
    forecast = model.predict(future)

    return forecast


yahoo_finance_tool = Tool(
    name="Yahoo Finance Tool",
    description="Fetches stocks prices for {ticket} from the last year about a specific company from Yahoo Finance API",
    func=lambda ticket: fetch_stock_price(ticket),
)
# IMPORTANDO OPENAI LLM - GPT
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
# os.getenv("OPENAI_API_SECRET")

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
)


stockPriceAnalyst = Agent(
    role="Senior stock price Analyst",
    goal="Find the {ticket} stock price and analyses trends",
    backstory="""You're highly experienced in analyzing the price of an specific stock
    and make predictions about its future price.""",
    verbose=True,
    llm=llm,
    max_iter=5,
    memory=True,
    tools=[yahoo_finance_tool],
    allow_delegation=False,
)

getStockPrice = Task(
    description="Analyze the stock {ticket} price history and create a trend analyses of up, down or sideways",
    expected_output="""" Specify the current trend stock price - up, down or sideways. 
    eg. stock= 'APPL, price UP'
""",
    agent=stockPriceAnalyst,
)

# IMPORTANT A TOOL DE SEARCH
search_tool = DuckDuckGoSearchResults(backend="news", num_results=10)

newsAnalyst = Agent(
    role="Stock News Analyst",
    goal="""Create a short summary of the market news related to the stock {ticket} company. Specify the current trend - up, down or sideways with
    the news context. For each request stock asset, specify a numbet between 0 and 100, where 0 is extreme fear and 100 is extreme greed.""",
    backstory="""You're highly experienced in analyzing the market trends and news and have tracked assest for more then 10 years.

    You're also master level analyts in the tradicional markets and have deep understanding of human psychology.

    You understand news, theirs tittles and information, but you look at those with a health dose of skepticism. 
    You consider also the source of the news articles. 
    """,
    verbose=True,
    llm=llm,
    max_iter=10,
    memory=True,
    tools=[search_tool],
    allow_delegation=False,
)

get_news = Task(
    description=f"""Take the stock and always include BTC to it (if not request).
    Use the search tool to search each one individually. 

    The current date is {datetime.now()}.

    Compose the results into a helpfull report""",
    expected_output=""""A summary of the overall market and one sentence summary for each request asset. 
    Include a fear/greed score for each asset based on the news. Use format:
    <STOCK ASSET>
    <SUMMARY BASED ON NEWS>
    <TREND PREDICTION>
    <FEAR/GREED SCORE>
""",
    agent=newsAnalyst,
)

stockAnalystWrite = Agent(
    role="Senior Stock Analyts Writer",
    goal=""""Analyze the trends price and news and write an insighfull compelling and informative 3 paragraph long newsletter based on the stock report and price trend. """,
    backstory="""You're widely accepted as the best stock analyst in the market. You understand complex concepts and create compelling stories
    and narratives that resonate with wider audiences. 

    You understand macro factors and combine multiple theories - eg. cycle theory and fundamental analyses. 
    You're able to hold multiple opinions when analyzing anything.
""",
    verbose=True,
    llm=llm,
    max_iter=5,
    memory=True,
    allow_delegation=True,
)

writeAnalyses = Task(
    description="""Use the stock price trend and the stock news report to create an analyses and write the newsletter about the {ticket} company
    that is brief and highlights the most important points.
    Focus on the stock price trend, news and fear/greed score. What are the near future considerations?
    Include the previous analyses of stock trend and news summary.
""",
    expected_output=""""An eloquent 3 paragraphs newsletter formated as markdown in an easy readable manner. It should contain:

    - 3 bullets executive summary 
    - Introduction - set the overall picture and spike up the interest
    - main part provides the meat of the analysis including the news summary and fead/greed scores
    - summary - key facts and concrete future trend prediction - up, down or sideways.
""",
    agent=stockAnalystWrite,
    context=[getStockPrice, get_news],
)

crew = Crew(
    agents=[stockPriceAnalyst, newsAnalyst, stockAnalystWrite],
    tasks=[getStockPrice, get_news, writeAnalyses],
    verbose=2,
    process=Process.hierarchical,
    full_output=True,
    share_crew=False,
    manager_llm=llm,
    max_iter=15,
)

# results = crew.kickoff(inputs={"ticket": "AAPL"})


# Função para extrair o Fear/Greed Score
def extract_fear_greed_score(final_output):
    # Divide o texto em linhas
    lines = final_output.split("\n")

    for line in lines:
        if "FEAR/GREED SCORE" in line:
            # Tenta extrair o número da linha
            score_str = line.split(":")[-1].strip()
            try:
                return int(score_str)
            except ValueError:
                pass
    return None


# Exemplo de uso
final_output = """
AAPL
Resumo baseado nas notícias...
TREND PREDICTION: UP
FEAR/GREED SCORE: 75
"""

score = extract_fear_greed_score(final_output)
print(score)  # Deve imprimir 75

with st.sidebar:
    st.header("Enter the Stock to Research")

    with st.form(key="research_form"):
        topic = st.text_input("Select the ticket")
        submit_button = st.form_submit_button(label="Run Research")
if submit_button:
    if not topic:
        st.error("Please fill the ticket field")
    else:
        results = crew.kickoff(inputs={"ticket": topic})

        st.subheader("Results of research:")
        st.write(results["final_output"])

        stock_data = fetch_stock_price(topic)

        if not stock_data.empty:
            # Interface do usuário usando Streamlit
            st.title("Previsão de Preços de Ações")
            st.subheader("Preço Histórico da Ação")
            st.line_chart(stock_data["Close"])

            st.subheader("Previsão de Preços Futuros")
            forecast = predict_stock_price(stock_data)
            st.line_chart(forecast[["ds", "yhat"]].set_index("ds"))
            # Extraia o Fear/Greed Score do resultado
            fear_greed_score = extract_fear_greed_score(results["final_output"])

            # Verifica se o score foi extraído com sucesso
            if fear_greed_score is not None:
                # Use st.feedback para mostrar o score
                st.feedback(
                    label=f"Fear/Greed Score for {topic}",
                    value=fear_greed_score,
                    min_value=0,
                    max_value=100,
                )
            else:
                st.error("Não foi possível extrair o Fear/Greed Score.")
        else:
            st.error("Não foi possível obter dados para o ticket fornecido.")
