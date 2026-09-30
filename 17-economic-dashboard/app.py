from io import StringIO
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

st.set_page_config(page_title="MacroScope",page_icon="🌐",layout="wide")
st.title("MacroScope — macroeconomic research dashboard")
st.caption("Explore sourced time series and statistical relationships without mistaking correlation for causation.")

SERIES={"Inflation index (CPI-U)":"CPIAUCSL","Unemployment rate":"UNRATE","Federal funds rate":"FEDFUNDS","Real GDP":"GDPC1","Industrial production":"INDPRO","Consumer sentiment":"UMCSENT"}
UNITS={"CPIAUCSL":"Index","UNRATE":"Percent","FEDFUNDS":"Percent","GDPC1":"Billions of chained dollars","INDPRO":"Index","UMCSENT":"Index"}


@st.cache_data(ttl=21600,show_spinner=False)
def fred_series(series_id):
    url=f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    response=requests.get(url,timeout=10);response.raise_for_status();d=pd.read_csv(StringIO(response.text));d.columns=["date",series_id];d.date=pd.to_datetime(d.date,errors="coerce");d[series_id]=pd.to_numeric(d[series_id],errors="coerce")
    return d.dropna(),url,datetime.now(timezone.utc).isoformat()


@st.cache_data
def synthetic():
    rng=np.random.default_rng(22);dates=pd.date_range("1990-01-01","2026-01-01",freq="MS");n=len(dates)
    inflation=np.exp(np.cumsum(.0022+rng.normal(0,.0025,n)))*130;unemp=np.clip(5.7+1.8*np.sin(np.arange(n)/43)+rng.normal(0,.35,n),2.8,13);rate=np.clip(3.2+2*np.sin(np.arange(n)/29)+rng.normal(0,.3,n),0,12)
    gdp=9000*np.exp(np.cumsum(.002+rng.normal(0,.003,n)));production=70*np.exp(np.cumsum(.001+rng.normal(0,.004,n)));sent=np.clip(88-3*(unemp-5.7)+rng.normal(0,5,n),45,115)
    return pd.DataFrame({"date":dates,"CPIAUCSL":inflation,"UNRATE":unemp,"FEDFUNDS":rate,"GDPC1":gdp,"INDPRO":production,"UMCSENT":sent})


labels=st.sidebar.multiselect("Indicators",list(SERIES),default=list(SERIES)[:3]);ids=[SERIES[x] for x in labels]
mode=st.sidebar.radio("Data mode",["FRED public data","Offline synthetic demo"])
if not ids:st.info("Select at least one indicator.");st.stop()
if mode=="FRED public data":
    pieces=[];meta=[]
    try:
        for sid in ids:
            d,url,fetched=fred_series(sid);pieces.append(d.set_index("date"));meta.append({"Series":sid,"Source":url,"Fetched (UTC)":fetched,"Last observation":d.date.max().date()})
        raw=pd.concat(pieces,axis=1,sort=False).sort_index();source="Federal Reserve Bank of St. Louis FRED CSV service"
    except Exception as exc:
        st.warning(f"FRED request failed ({exc}). Showing clearly labeled synthetic fallback.");raw=synthetic().set_index("date")[ids];meta=[];source="Synthetic fallback"
else:raw=synthetic().set_index("date")[ids];meta=[];source="Synthetic macroeconomic demonstration series"
start,end=st.sidebar.slider("Year range",int(raw.index.year.min()),int(raw.index.year.max()),(max(int(raw.index.year.min()),2000),int(raw.index.year.max())))
data=raw[raw.index.year.to_series().between(start,end).to_numpy()].copy();monthly=data.resample("MS").mean().interpolate(limit=3)
tabs=st.tabs(["Levels","Changes","Correlations","Lead/lag explorer","Sources & concepts"])
with tabs[0]:
    st.info(source);long=data.reset_index().melt("date",var_name="Series",value_name="Value");st.plotly_chart(px.line(long,x="date",y="Value",facet_row="Series",title="Raw values (separate scales)",height=260+190*len(ids)),width="stretch")
    normalized=data.apply(lambda x:x/x.dropna().iloc[0]*100);st.plotly_chart(px.line(normalized,title="Normalized index (first visible observation = 100)"),width="stretch")
with tabs[1]:
    change=st.radio("Change",["Month over month","Year over year"],horizontal=True);periods=1 if change.startswith("Month") else 12;chg=monthly.pct_change(periods)*100
    st.plotly_chart(px.line(chg,title=f"{change} percent change"),width="stretch");st.caption("Percent changes are not always the best transformation for rate series; interpret interest and unemployment rate changes carefully.")
with tabs[2]:
    transformed=monthly.pct_change(12).dropna();st.plotly_chart(px.imshow(transformed.corr(),text_auto=".2f",zmin=-1,zmax=1,color_continuous_scale="RdBu_r",title="Correlation of 12-month percent changes"),width="stretch")
    st.caption("Correlation describes co-movement in this period and transformation; it does not identify causal effects.")
with tabs[3]:
    if len(ids)<2:st.info("Select at least two indicators.")
    else:
        a=st.selectbox("Indicator A",ids);b=st.selectbox("Indicator B",[x for x in ids if x!=a]);base=monthly[[a,b]].pct_change(12).dropna();rows=[]
        for lag in range(-24,25):rows.append({"Lag months for B":lag,"Correlation":base[a].corr(base[b].shift(lag))})
        lead=pd.DataFrame(rows);best=lead.iloc[lead.Correlation.abs().argmax()];st.plotly_chart(px.line(lead,x="Lag months for B",y="Correlation",markers=True,title="Exploratory lead/lag correlation"),width="stretch")
        st.write(f"Largest absolute sample correlation: {best.Correlation:.2f} at lag {int(best['Lag months for B'])} months.")
        st.caption("Multiple testing, autocorrelation, revisions, shared trends, and third variables can create apparent leads. This is not causal evidence.")
with tabs[4]:
    if meta:st.dataframe(pd.DataFrame(meta),width="stretch",hide_index=True)
    st.markdown("""**Inflation** is the rate of change in a price index, not the price level itself. **Unemployment** measures people without work who meet the survey definition and are actively seeking work. **Real GDP** adjusts output for price changes; nominal GDP does not. **Interest rates** are percentages over a period, not price indexes.

Economic series have different frequencies, release lags, revisions, seasonal adjustments, and missing observations. This app retains raw observations for level charts and uses monthly means plus limited interpolation for cross-series comparisons. Verify each FRED series' detailed notes before substantive use.""")
