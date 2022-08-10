import json
from datetime import date
from urllib.request import urlopen
import time

import altair as alt
import numpy as np
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
from pandas.io.json import json_normalize
import hydralit_components as hc
from streamlit_lottie import st_lottie
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

today = date.today()

st.set_page_config(
    page_title="Coviz: COVID19 Tracking Application",
    page_icon= "✅",
    layout='wide'
)

#defining lottie function to visualize animated pictures
def load_lottiefile(filepath: str):
    with open(filepath) as f:
        return json.load(f)


@st.cache(ttl=3*60*60, suppress_st_warning=True)
def get_data():
    US_confirmed = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv'
    US_deaths = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv'
    confirmed = pd.read_csv(US_confirmed)
    deaths = pd.read_csv(US_deaths)
    return confirmed, deaths

confirmed, deaths = get_data()
FIPSs = confirmed.groupby(['Province_State', 'Admin2']).FIPS.unique().apply(pd.Series).reset_index()
FIPSs.columns = ['State', 'County', 'FIPS']
FIPSs['FIPS'].fillna(0, inplace = True)
FIPSs['FIPS'] = FIPSs.FIPS.astype(int).astype(str).str.zfill(5)

@st.cache(ttl=3*60*60, suppress_st_warning=True)
def get_testing_data(County):
    apiKey = '9fe19182c5bf4d1bb105da08e593a578'
    if len(County) == 1:
        #print(len(County))
        f = FIPSs[FIPSs.County == County[0]].FIPS.values[0]
        #print(f)
        path1 = 'https://data.covidactnow.org/latest/us/counties/'+f+'.OBSERVED_INTERVENTION.timeseries.json?apiKey='+apiKey
        #print(path1)
        df = json.loads(requests.get(path1).text)
        #print(df.keys())
        data = pd.DataFrame.from_dict(df['actualsTimeseries'])
        data['Date'] = pd.to_datetime(data['date'])
        data = data.set_index('Date')
        #print(data.tail())
        try:
            data['new_negative_tests'] = data['cumulativeNegativeTests'].diff()
            data.loc[(data['new_negative_tests'] < 0)] = np.nan
        except: 
            data['new_negative_tests'] = np.nan
            st.text('Negative test data not avilable')
        data['new_negative_tests_rolling'] = data['new_negative_tests'].fillna(0).rolling(14).mean()


        try:
            data['new_positive_tests'] = data['cumulativePositiveTests'].diff()
            data.loc[(data['new_positive_tests'] < 0)] = np.nan
        except: 
            data['new_positive_tests'] = np.nan
            st.text('test data not avilable')
        data['new_positive_tests_rolling'] = data['new_positive_tests'].fillna(0).rolling(14).mean()
        data['new_tests'] = data['new_negative_tests']+data['new_positive_tests']
        data['new_tests_rolling'] = data['new_tests'].fillna(0).rolling(14).mean()
        data['testing_positivity_rolling'] = (data['new_positive_tests_rolling'] / data['new_tests_rolling'])*100
        #data['testing_positivity_rolling'].tail(14).plot()
        #plt.show()
        return data['new_tests_rolling'], data['testing_positivity_rolling'].iloc[-1:].values[0]
    elif (len(County) > 1) & (len(County) < 5):
        new_positive_tests = []
        new_negative_tests = []
        new_tests = []
        for c in County:
            f = FIPSs[FIPSs.County == c].FIPS.values[0]
            path1 = 'https://data.covidactnow.org/latest/us/counties/'+f+'.OBSERVED_INTERVENTION.timeseries.json?apiKey='+apiKey
            df = json.loads(requests.get(path1).text)
            data = pd.DataFrame.from_dict(df['actualsTimeseries'])
            data['Date'] = pd.to_datetime(data['date'])
            data = data.set_index('Date')
            try:
                data['new_negative_tests'] = data['cumulativeNegativeTests'].diff()
                data.loc[(data['new_negative_tests'] < 0)] = np.nan
            except: 
                data['new_negative_tests'] = np.nan
                #print('Negative test data not avilable')

            try:
                data['new_positive_tests'] = data['cumulativePositiveTests'].diff()
                data.loc[(data['new_positive_tests'] < 0)] = np.nan
            except: 
                data['new_positive_tests'] = np.nan
                #print('Negative test data not avilable')
            data['new_tests'] = data['new_negative_tests']+data['new_positive_tests']

            new_positive_tests.append(data['new_positive_tests'])
            #new_negative_tests.append(data['new_tests'])
            new_tests.append(data['new_tests'])
            #print(data.head())

        new_positive_tests_rolling = pd.concat(new_positive_tests, axis = 1).sum(axis = 1)
        new_positive_tests_rolling = new_positive_tests_rolling.fillna(0).rolling(14).mean()
        #print('new test merging of counties')
        #print(pd.concat(new_tests, axis = 1).head().sum(axis = 1))
        new_tests_rolling = pd.concat(new_tests, axis = 1).sum(axis = 1)
        new_tests_rolling = new_tests_rolling.fillna(0).rolling(14).mean()
        new_tests_rolling = pd.DataFrame(new_tests_rolling).fillna(0)
        new_tests_rolling.columns = ['new_tests_rolling']
        #print('whole df')
        #print(type(new_tests_rolling))
        #print(new_tests_rolling.head())
        #print('single column')
        #print(new_tests_rolling['new_tests_rolling'].head())
        #print('new_positive_tests_rolling')
        #print(new_positive_tests_rolling.head())
        #print('new_tests_rolling')
        #print(new_tests_rolling.head())
        data_to_show = (new_positive_tests_rolling / new_tests_rolling.new_tests_rolling)*100
        #print(data_to_show.shape)
        #print(data_to_show.head())
        #print(data_to_show.columns)
        #print(data_to_show.iloc[-1:].values[0])
        return new_tests_rolling, data_to_show.iloc[-1:].values[0]
    else:
        st.text('Getting testing data for California State')
        path1 = 'https://data.covidactnow.org/latest/us/states/CA.OBSERVED_INTERVENTION.timeseries.json'
        df = json.loads(requests.get(path1).text)
        data = pd.DataFrame.from_dict(df['actualsTimeseries'])
        data['Date'] = pd.to_datetime(data['date'])
        data = data.set_index('Date')

        try:
            data['new_negative_tests'] = data['cumulativeNegativeTests'].diff()
            data.loc[(data['new_negative_tests'] < 0)] = np.nan
        except:
            data['new_negative_tests'] = np.nan
            print('Negative test data not available')
        data['new_negative_tests_rolling'] = data['new_negative_tests'].fillna(0).rolling(14).mean()


        try:
            data['new_positive_tests'] = data['cumulativePositiveTests'].diff()
            data.loc[(data['new_positive_tests'] < 0)] = np.nan
        except:
            data['new_positive_tests'] = np.nan
            st.text('test data not available')
        data['new_positive_tests_rolling'] = data['new_positive_tests'].fillna(0).rolling(14).mean()
        data['new_tests'] = data['new_negative_tests']+data['new_positive_tests']
        data['new_tests_rolling'] = data['new_tests'].fillna(0).rolling(14).mean()
        data['testing_positivity_rolling'] = (data['new_positive_tests_rolling'] / data['new_tests_rolling'])*100
        return data['new_tests_rolling'], data['testing_positivity_rolling'].iloc[-1:].values[0]


def plot_county(county):
    testing_df, testing_percent = get_testing_data(County=county)
    #print(testing_df.head())
    county_confirmed = confirmed[confirmed.Admin2.isin(county)]
    county_confirmed_time = county_confirmed.drop(county_confirmed.iloc[:, 0:12], axis=1).T
    county_confirmed_time = county_confirmed_time.sum(axis= 1)
    county_confirmed_time = county_confirmed_time.reset_index()
    county_confirmed_time.columns = ['date', 'cases']
    county_confirmed_time['Datetime'] = pd.to_datetime(county_confirmed_time['date'])
    county_confirmed_time = county_confirmed_time.set_index('Datetime')
    del county_confirmed_time['date']
    incidence= pd.DataFrame(county_confirmed_time.cases.diff())
    incidence.columns = ['incidence']
    chart_max = incidence.max().values[0]+500

    county_deaths = deaths[deaths.Admin2.isin(county)]
    population = county_deaths.Population.values.sum()

    del county_deaths['Population']
    county_deaths_time = county_deaths.drop(county_deaths.iloc[:, 0:11], axis=1).T
    county_deaths_time = county_deaths_time.sum(axis= 1)

    county_deaths_time = county_deaths_time.reset_index()
    county_deaths_time.columns = ['date', 'deaths']
    county_deaths_time['Datetime'] = pd.to_datetime(county_deaths_time['date'])
    county_deaths_time = county_deaths_time.set_index('Datetime')
    del county_deaths_time['date']

    cases_per100k  = ((county_confirmed_time) * 100000 / population)
    cases_per100k.columns = ['cases per 100K']
    cases_per100k['rolling average'] = cases_per100k['cases per 100K'].rolling(7).mean()

    deaths_per100k  = ((county_deaths_time) * 100000 / population)
    deaths_per100k.columns = ['deaths per 100K']
    deaths_per100k['rolling average'] = deaths_per100k['deaths per 100K'].rolling(7).mean()


    incidence['rolling_incidence'] = incidence.incidence.rolling(7).mean()
    metric = (incidence['rolling_incidence'] * 100000 / population).iloc[[-1]]

    if len(county) == 1:
        st.subheader('Current situation of COVID-19 cases in '+', '.join(map(str, county))+' county ('+ str(today)+')')
        column1, column2, column3, column4 = st.columns(4)
        number1 = int(metric.values[0])
        column1.metric('New cases averaged over last 7 days', number1)
        column2.metric("Population under consideration", population)
        column3.metric("Total cases", county_confirmed_time.tail(1).values[0][0])
        column4.metric("Total deaths", county_deaths_time.tail(1).values[0][0])
    else:
        st.subheader('Current situation of COVID-19 cases in '+', '.join(map(str, county))+' counties ('+ str(today)+')')
        column1, column2, column3, column4 = st.columns(4)
        number1 = int(metric.values[0])
        column1.metric("New cases averaged over last 7 days", number1)
        column2.metric("Population under consideration", population)
        column3.metric('Total cases', county_confirmed_time.tail(1).values[0][0])
        column4.metric("Total deaths", county_deaths_time.tail(1).values[0][0])
    c1 = st.container()
    c2 = st.container()
    c3 = st.container()

    if len(county)==1:
        C = county[0]
        with c2:
            a1, _, a2 = st.columns((3.9, 0.2, 3.9))     
            with a1:
                f = FIPSs[FIPSs.County == C].FIPS.values[0]
                components.iframe("https://covidactnow.org/embed/us/county/"+f, width=350, height=365, scrolling=False)
                
            with a2:
                st.markdown('')
    elif len(county) <= 3:
        with c2:
            st.write('')
            st.write('')
            
        with c3:
            columns = st.columns(len(county))
            for idx, C in enumerate(county):
                with columns[idx]:
                    st.write('')
                    st.write('')
                    f = FIPSs[FIPSs.County == C].FIPS.values[0]
                    components.iframe("https://covidactnow.org/embed/us/county/"+f, width=350, height=365, scrolling=False)

    ### Experiment with Altair instead of Matplotlib.
    with c1:
        a2, _, a1 = st.columns((3.9, 0.2, 3.9))

        incidence = incidence.reset_index()
        incidence['nomalized_rolling_incidence'] = incidence['rolling_incidence'] * 100000 / population
        incidence['Phase 2 Threshold'] = 25
        incidence['Phase 3 Threshold'] = 10
        scale = alt.Scale(
            domain=[
                "rolling_incidence",
                "Phase 2 Threshold",
                "Phase 3 Threshold"
            ], range=['#377eb8', '#e41a1c', '#4daf4a'])
        base = alt.Chart(
            incidence,
            title='(A) Weekly rolling mean of incidence per 100K'
        ).transform_calculate(
            base_="'rolling_incidence'",
            phase2_="'Phase 2 Threshold'",
            phase3_="'Phase 3 Threshold'",
        )
        
        ax4 = base.mark_line(strokeWidth=3).encode(
            x=alt.X("Datetime", axis = alt.Axis(title='Date')),
            y=alt.Y("nomalized_rolling_incidence", axis=alt.Axis(title='per 100 thousand')),
            color=alt.Color("base_:N", scale=scale, title="")
        )

        line1 = base.mark_line(strokeDash=[8, 8], strokeWidth=2).encode(
            x=alt.X("Datetime", axis=alt.Axis(title = 'Date')),
            y=alt.Y("Phase 2 Threshold", axis=alt.Axis(title='Count')),
            color=alt.Color("phase2_:N", scale=scale, title="")
        )

        line2 = base.mark_line(strokeDash=[8, 8], strokeWidth=2).encode(
            x=alt.X("Datetime", axis=alt.Axis(title='Date')),
            y=alt.Y("Phase 3 Threshold", axis=alt.Axis(title='Count')),
            color=alt.Color("phase3_:N", scale=scale, title="")
        )

        with a2:
            st.altair_chart(ax4 + line1 + line2, use_container_width=True)

        ax3 = alt.Chart(incidence, title = '(B) Daily incidence (new cases)').mark_bar().encode(
            x=alt.X("Datetime",axis = alt.Axis(title = 'Date')),
            y=alt.Y("incidence",axis = alt.Axis(title = 'Incidence'), scale=alt.Scale(domain=(0, chart_max), clamp=True))
        )
        
        with a1:
            st.altair_chart(ax3, use_container_width=True)
        
        a3, _, a4 = st.columns((3.9, 0.2, 3.9))
        testing_df = pd.DataFrame(testing_df).reset_index()
        #print(testing_df.head())
        #print(type(testing_df))
        
        base = alt.Chart(testing_df, title = '(D) Daily new tests').mark_line(strokeWidth=3).encode(
            x=alt.X("Date",axis = alt.Axis(title = 'Date')),
            y=alt.Y("new_tests_rolling",axis = alt.Axis(title = 'Daily new tests'))
        )
        with a4:
            st.altair_chart(base, use_container_width=True)

        county_confirmed_time = county_confirmed_time.reset_index()
        county_deaths_time = county_deaths_time.reset_index()
        cases_and_deaths = county_confirmed_time.set_index("Datetime").join(county_deaths_time.set_index("Datetime"))
        cases_and_deaths = cases_and_deaths.reset_index()

        # Custom colors for layered charts.
        # See https://stackoverflow.com/questions/61543503/add-legend-to-line-bars-to-altair-chart-without-using-size-color.
        scale = alt.Scale(domain=["cases", "deaths"], range=['#377eb8', '#e41a1c'])
        base = alt.Chart(
            cases_and_deaths,
            title='(C) Cumulative cases and deaths'
        ).transform_calculate(
            cases_="'cases'",
            deaths_="'deaths'",
        )

        c = base.mark_line(strokeWidth=3).encode(
            x=alt.X("Datetime", axis=alt.Axis(title = 'Date')),
            y=alt.Y("cases", axis=alt.Axis(title = 'Count')),
            color=alt.Color("cases_:N", scale=scale, title="")
        )

        d = base.mark_line(strokeWidth=3).encode(
            x=alt.X("Datetime", axis=alt.Axis(title='Date')),
            y=alt.Y("deaths", axis=alt.Axis(title = 'Count')),
            color=alt.Color("deaths_:N", scale=scale, title="")
        )
        with a3:
            st.altair_chart(c+d, use_container_width=True)


def plot_state():
    def get_testing_data_state():
            st.text('Getting testing data for California State')
            path1 = 'https://data.covidactnow.org/latest/us/states/CA.OBSERVED_INTERVENTION.timeseries.json'
            df = json.loads(requests.get(path1).text)
            data = pd.DataFrame.from_dict(df['actualsTimeseries'])
            data['Date'] = pd.to_datetime(data['date'])
            data = data.set_index('Date')

            try:
                data['new_negative_tests'] = data['cumulativeNegativeTests'].diff()
                data.loc[(data['new_negative_tests'] < 0)] = np.nan
            except:
                data['new_negative_tests'] = np.nan
                print('Negative test data not available')
            data['new_negative_tests_rolling'] = data['new_negative_tests'].fillna(0).rolling(14).mean()


            try:
                data['new_positive_tests'] = data['cumulativePositiveTests'].diff()
                data.loc[(data['new_positive_tests'] < 0)] = np.nan
            except:
                data['new_positive_tests'] = np.nan
                st.text('test data not available')
            data['new_positive_tests_rolling'] = data['new_positive_tests'].fillna(0).rolling(14).mean()
            data['new_tests'] = data['new_negative_tests']+data['new_positive_tests']
            data['new_tests_rolling'] = data['new_tests'].fillna(0).rolling(14).mean()
            data['testing_positivity_rolling'] = (data['new_positive_tests_rolling'] / data['new_tests_rolling'])*100
            # return data['new_tests_rolling'], data['testing_positivity_rolling'].iloc[-1:].values[0]
            testing_df, testing_percent = data['new_tests_rolling'], data['testing_positivity_rolling'].iloc[-1:].values[0]
            county_confirmed = confirmed[confirmed.Province_State == 'California']
            #county_confirmed = confirmed[confirmed.Admin2 == county]
            county_confirmed_time = county_confirmed.drop(county_confirmed.iloc[:, 0:12], axis=1).T #inplace=True, axis=1
            county_confirmed_time = county_confirmed_time.sum(axis= 1)
            county_confirmed_time = county_confirmed_time.reset_index()
            county_confirmed_time.columns = ['date', 'cases']
            county_confirmed_time['Datetime'] = pd.to_datetime(county_confirmed_time['date'])
            county_confirmed_time = county_confirmed_time.set_index('Datetime')
            del county_confirmed_time['date']
            #print(county_confirmed_time.head())
            incidence = pd.DataFrame(county_confirmed_time.cases.diff())
            incidence.columns = ['incidence']

            #temp_df_time = temp_df.drop(['date'], axis=0).T #inplace=True, axis=1
            county_deaths = deaths[deaths.Province_State == 'California']
            population = county_deaths.Population.values.sum()

            del county_deaths['Population']
            county_deaths_time = county_deaths.drop(county_deaths.iloc[:, 0:11], axis=1).T #inplace=True, axis=1
            county_deaths_time = county_deaths_time.sum(axis= 1)

            county_deaths_time = county_deaths_time.reset_index()
            county_deaths_time.columns = ['date', 'deaths']
            county_deaths_time['Datetime'] = pd.to_datetime(county_deaths_time['date'])
            county_deaths_time = county_deaths_time.set_index('Datetime')
            del county_deaths_time['date']

            cases_per100k  = ((county_confirmed_time)*100000/population)
            cases_per100k.columns = ['cases per 100K']
            cases_per100k['rolling average'] = cases_per100k['cases per 100K'].rolling(7).mean()

            deaths_per100k  = ((county_deaths_time)*100000/population)
            deaths_per100k.columns = ['deaths per 100K']
            deaths_per100k['rolling average'] = deaths_per100k['deaths per 100K'].rolling(7).mean()

            incidence['rolling_incidence'] = incidence.incidence.rolling(7).mean()
            return population, testing_df, testing_percent, county_deaths_time, county_confirmed_time, incidence
    # metric = (incidence['rolling_incidence']*100000/population).iloc[[-1]]

    #print(county_deaths_time.tail(1).values[0])
    #print(cases_per100k.head())
    population, testing_df, testing_percent, county_deaths_time, county_confirmed_time, incidence = get_testing_data_state()
    st.subheader('Current situation of COVID-19 cases in California ('+ str(today)+')')
    column1, column2, column3, column4 = st.columns(4)
    number2 = int(testing_percent)
    number3 = str(number2)+"%"
    column1.metric("% test positivity (14 day average)", number3)
    column2.metric("Population under consideration", population)
    column3.metric('Total cases', county_confirmed_time.tail(1).values[0][0])
    column4.metric("Total deaths", county_deaths_time.tail(1).values[0][0])
    c1 = st.container()
    c2 = st.container()
    c3 = st.container()

    with c2:
        a1, _, a2 = st.columns([0.5,1,0.5])     
        with a1:
            #f = FIPSs[FIPSs.County == C].FIPS.values[0]
            st.markdown("")

        with _:
            components.iframe("https://covidactnow.org/embed/us/california-ca", width=350, height=365, scrolling=False)
            st.markdown("")
            
    ### Experiment with Altair instead of Matplotlib.
    with c1:
        a2, _, a1 = st.columns((3.9, 0.2, 3.9))

        incidence = incidence.reset_index()
        incidence['nomalized_rolling_incidence'] = incidence['rolling_incidence'] * 100000 / population
        incidence['Phase 2 Threshold'] = 25
        incidence['Phase 3 Threshold'] = 10
        
        scale = alt.Scale(
            domain=[
                "rolling_incidence",
                "Phase 2 Threshold",
                "Phase 3 Threshold"
            ], range=['#377eb8', '#e41a1c', '#4daf4a'])
        base = alt.Chart(
            incidence,
            title='(A) Weekly rolling mean of incidence per 100K'
        ).transform_calculate(
            base_="'rolling_incidence'",
            phase2_="'Phase 2 Threshold'",
            phase3_="'Phase 3 Threshold'",
        )
        
        ax4 = base.mark_line(strokeWidth=3).encode(
            x=alt.X("Datetime", axis = alt.Axis(title='Date')),
            y=alt.Y("nomalized_rolling_incidence", axis=alt.Axis(title='per 100 thousand')),
            color=alt.Color("base_:N", scale=scale, title="")
        )

        line1 = base.mark_line(strokeDash=[8, 8], strokeWidth=2).encode(
            x=alt.X("Datetime", axis=alt.Axis(title = 'Date')),
            y=alt.Y("Phase 2 Threshold", axis=alt.Axis(title='Count')),
            color=alt.Color("phase2_:N", scale=scale, title="")
        )

        line2 = base.mark_line(strokeDash=[8, 8], strokeWidth=2).encode(
            x=alt.X("Datetime", axis=alt.Axis(title='Date')),
            y=alt.Y("Phase 3 Threshold", axis=alt.Axis(title='Count')),
            color=alt.Color("phase3_:N", scale=scale, title="")
        )
        with a2:
            st.altair_chart(ax4 + line1 + line2, use_container_width=True)

        ax3 = alt.Chart(incidence, title = '(B) Daily incidence (new cases)').mark_bar().encode(
            x=alt.X("Datetime",axis = alt.Axis(title = 'Date')),
            y=alt.Y("incidence",axis = alt.Axis(title = 'Incidence'))
        )
        
        with a1:
            st.altair_chart(ax3, use_container_width=True)
        
        a3, _, a4 = st.columns((3.9, 0.2, 3.9))
        testing_df = pd.DataFrame(testing_df).reset_index()
        #print(testing_df.head())
        #print(type(testing_df))
        
        base = alt.Chart(testing_df, title = '(D) Daily new tests').mark_line(strokeWidth=3).encode(
            x=alt.X("Date",axis = alt.Axis(title = 'Date')),
            y=alt.Y("new_tests_rolling",axis = alt.Axis(title = 'Daily new tests'))
        )
        with a4:
            st.altair_chart(base, use_container_width=True)

        county_confirmed_time = county_confirmed_time.reset_index()
        county_deaths_time = county_deaths_time.reset_index()
        cases_and_deaths = county_confirmed_time.set_index("Datetime").join(county_deaths_time.set_index("Datetime"))
        cases_and_deaths = cases_and_deaths.reset_index()

        # Custom colors for layered charts.
        # See https://stackoverflow.com/questions/61543503/add-legend-to-line-bars-to-altair-chart-without-using-size-color.
        scale = alt.Scale(domain=["cases", "deaths"], range=['#377eb8', '#e41a1c'])
        base = alt.Chart(
            cases_and_deaths,
            title='(C) Cumulative cases and deaths'
        ).transform_calculate(
            cases_="'cases'",
            deaths_="'deaths'",
        )

        c = base.mark_line(strokeWidth=3).encode(
            x=alt.X("Datetime", axis=alt.Axis(title = 'Date')),
            y=alt.Y("cases", axis=alt.Axis(title = 'Count')),
            color=alt.Color("cases_:N", scale=scale, title="")
        )

        d = base.mark_line(strokeWidth=3).encode(
            x=alt.X("Datetime", axis=alt.Axis(title='Date')),
            y=alt.Y("deaths", axis=alt.Axis(title = 'Count')),
            color=alt.Color("deaths_:N", scale=scale, title="")
        )
        with a3:
            st.altair_chart(c+d, use_container_width=True)


## functions end here, title, sidebar setting and descriptions start here
t1, t2 = st.columns(2)
with t1:
    st.markdown('# COVID-19 Dashboard 😷')

with t2:
    st.write("")
    st.write("")
    st.write("""
    **Health Care Analytics** | Made with ❤️ by Ali Maatouk 
    \n 🟢 Powered by MSBA - OSB @ AUB
    """)


menu_data = [
    {'label': "Counties", 'icon': 'bi bi-bar-chart-line'},
    {'label': 'California', 'icon': '🇺🇸'},
    {'label': 'Vaccines', 'icon': '💉'},
    {'label': 'Health Equity', 'icon': '⚖️'},
    {'label':"Overview", 'icon':'🔍'}]

over_theme = {'txc_inactive': 'white','menu_background':'rgb(0,0,128)', 'option_active':'white'}

menu_id = hc.nav_bar(
    menu_definition=menu_data,
    override_theme=over_theme,
    hide_streamlit_markers=True,
    sticky_nav=True, #at the top or not
    sticky_mode='sticky', #jumpy or not-jumpy, but sticky or pinned
)

if menu_id == 'Counties':
    st.markdown('## Select counties of interest')
    CA_counties = confirmed[confirmed.Province_State == 'California'].Admin2.unique().tolist()
    counties = st.multiselect('', CA_counties, default=['Yolo', 'Solano', 'Sacramento'])
    # Limit to the first 5 counties.
    counties = counties[:5]
    if not counties:
        # If no counties are specified, just plot the state.
        st.markdown('> No counties were selected, falling back to showing statistics for California state.')
        plot_state()
    else:
        # Plot the aggregate and per-county details.
        plot_county(counties)
        for c in counties:
            st.write('')
            with st.expander(f"Expand for {c} County Details"):
                plot_county([c])
if menu_id == 'California':
    plot_state()

if menu_id == "Vaccines":
    colll1,colll2,colll3, colll4, colll5 = st.columns(5)
    coll1, coll2, coll3 = st.columns([1,10,1])
    
    with colll1:
        st.write("")
        lottie_vaccine= load_lottiefile("vaccine.json")
        st_lottie(lottie_vaccine, height=150, width=200)
    
    with colll2:
        lottie_vaccine2= load_lottiefile("vaccine2.json")
        st_lottie(lottie_vaccine2, height=150, width=200)
    
    with colll3:
        lottie_vaccine3= load_lottiefile("vaccine3.json")
        st_lottie(lottie_vaccine3, height=150, width=200)
    
    with colll4:
        lottie_vaccine4= load_lottiefile("vaccine4.json")
        st_lottie(lottie_vaccine4, height=150, width=200)

    with colll5:
        lottie_vaccine5= load_lottiefile("covid.json")
        st_lottie(lottie_vaccine5, height=150, width=200)    
    
    with coll2:
        def main():

            html_temp = """
        <div class='tableauPlaceholder' id='viz1657018560306' style='position: relative'><noscript><a href='#'><img alt='Vaccine ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;CO&#47;COVID-19VaccineDashboardPublicv2&#47;Vaccine&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='COVID-19VaccineDashboardPublicv2&#47;Vaccine' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;CO&#47;COVID-19VaccineDashboardPublicv2&#47;Vaccine&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-GB' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1657018560306');                    var vizElement = divElement.getElementsByTagName('object')[0];                    if ( divElement.offsetWidth > 800 ) { vizElement.style.minWidth='500px';vizElement.style.maxWidth='800px';vizElement.style.width='100%';vizElement.style.height='527px';} else if ( divElement.offsetWidth > 500 ) { vizElement.style.minWidth='500px';vizElement.style.maxWidth='800px';vizElement.style.width='100%';vizElement.style.height='527px';} else { vizElement.style.width='100%';vizElement.style.height='827px';}                     var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>"""
        
            st.components.v1.html(html_temp, height=550, scrolling=False)

        if __name__ == "__main__":
            main()

if menu_id == 'Health Equity':
    st.markdown("<h3 style='text-align: center; color: blue;'>The disparities in California diverse communities are severe</h1>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        #animation1 about latino dispersion
        lottie_latino= load_lottiefile("latino.json")
        st_lottie(lottie_latino)
        st.caption(f"""
            <div>
                <div style="vertical-align:left;font-size:16px;padding-left:5px;padding-top:5px;margin-left:0em";>
                 
Death rate for Latino people is 13% higher than statewide

Deaths per 100K people:

254 Latino

224 all ethnicities </div>""",unsafe_allow_html = True)

    with col2:
        #animation2 about Native American category
        lottie_na= load_lottiefile("native american.json")
        st_lottie(lottie_na)
        st.caption(f"""
            <div>
            <div style="vertical-align:left;font-size:16px;padding-left:5px;padding-top:5px;margin-left:0em";>

Case rate for Pacific Islanders is 81% higher than statewide

Cases per 100K people:

41,982 NHPI

23,154 all ethnicities
</div>""",unsafe_allow_html = True)

    with col3:
        #animation3 about black americans
        lottie_black= load_lottiefile("african american.json")
        st_lottie(lottie_black)
        st.caption(f"""
            <div>
                <div style="vertical-align:left;font-size:16px;padding-left:5px;padding-top:5px;margin-left:0em";>
                
Death rate for Black people is 18% higher than statewide

Deaths per 100K people:

266 Black

224 all ethnicities

</div>""",unsafe_allow_html = True)

    with col4:
        #animation4 about low income
        st.write("")
        st.write("")
        st.write("")
        lottie_poor= load_lottiefile("poor person.json")
        st_lottie(lottie_poor)
        st.write("")
        st.write("")
        st.caption(f"""
            <div>
                <div style="vertical-align:left;font-size:16px;padding-left:5px;padding-top:5px;margin-left:0em";>
                
Case rate for communities with median income <$40K is 20% higher than statewide

Cases per 100K people:

27,718 income <$40K

23,154 all income brackets

</div>""",unsafe_allow_html = True)

    st.markdown("data source: https://covid19.ca.gov/equity/")
    st.markdown("<h3 style='text-align: center; color: Green;'> COVID-19 Age, Race and Ethnicity Data </h1>", unsafe_allow_html=True)
    st.markdown("data source: https://www.cdph.ca.gov/Programs/CID/DCDC/Pages/COVID-19/Age-Race-Ethnicity.aspx")
    chosen_demo = st.selectbox('Please choose the demographic type:', ('', 'Cases and Deaths Associated with COVID-19 by Age Group in California', 'All Cases and Deaths associated with COVID-19 by Race and Ethnicity'))
    if chosen_demo == '':
        st.write("")
    if chosen_demo == "Cases and Deaths Associated with COVID-19 by Age Group in California":
        url = "https://docs.google.com/spreadsheets/d/1NbqgjOYCWq0eoBjBmRqvL40Zkx7F97yL1fPAQ4fTcHo/edit#gid=0"
        path = url.replace('/edit#gid=', '/export?format=csv&gid=')
        c_d_age = pd.read_csv(path, encoding="utf8")
        figure1 = make_subplots(rows=1, cols=2)
        figure1.add_trace(go.Bar(name='Cases', x=c_d_age["Age Group"], y=c_d_age["No. Cases"]), row=1, col=1)
        figure1.add_trace(go.Bar(name='Deaths', x=c_d_age["Age Group"], y=c_d_age["No. Deaths"]), row=1, col=2)
        figure1.update_layout(title_text="Cases and Deaths Associated with COVID-19 by Age Group in California ", width=1100)
        st.plotly_chart(figure1)
    if chosen_demo == "All Cases and Deaths associated with COVID-19 by Race and Ethnicity":
        url1 = "https://docs.google.com/spreadsheets/d/110WlCTn9CSWgf6_Ip4oAjyX4A8AfnXDzRxt1_8PkrYo/edit#gid=0"
        path11 = url1.replace('/edit#gid=', '/export?format=csv&gid=')
        c_d_race = pd.read_csv(path11, encoding="utf8")
        figure2 = make_subplots(rows=1, cols=2)
        figure2.add_trace( go.Bar(name='Cases', x=c_d_race["Race/Ethnicity"], y=c_d_race["No. Cases"]), row=1, col=1 )
        figure2.add_trace( go.Bar(name='Deaths', x=c_d_race["Race/Ethnicity"], y=c_d_race["No. Deaths"]), row=1, col=2)
        figure2.update_layout(title_text="All Cases and Deaths Associated with COVID-19 by Race and Ethnicity in California", width=1100)
        st.plotly_chart(figure2)
    chosen_age_group = st.selectbox('Please choose the age group:', ('', 'Proportions of Cases and Deaths by Race and Ethnicity Among Ages 0‐17', 'Proportions of Cases and Deaths by Race and Ethnicity Among Ages 18‐34', "Proportions of Cases and Deaths by Race and Ethnicity Among Ages 35‐49", "Proportions of Cases and Deaths by Race and Ethnicity Among Ages 50‐64", "Proportions of Cases and Deaths by Race and Ethnicity Among Ages 65-79", "Proportions of Cases and Deaths by Race and Ethnicity Among Ages 80+"))
    if chosen_age_group == '':
        st.write("")
    if chosen_age_group == 'Proportions of Cases and Deaths by Race and Ethnicity Among Ages 0‐17':
        url2 = "https://docs.google.com/spreadsheets/d/1ZixsfwtLh7c8Tfqgrro_W3YRL8jYfyMu6E034fnaDbo/edit#gid=0"
        path2 = url2.replace('/edit#gid=', '/export?format=csv&gid=')
        c_d_race_17 = pd.read_csv(path2, encoding="utf8")
        figure3 = make_subplots(rows=1, cols=2)
        figure3.add_trace(go.Bar(name='Cases', x=c_d_race_17["Race/Ethnicity"], y=c_d_race_17["No. Cases"]), row=1, col=1)
        figure3.add_trace( go.Bar(name='Deaths', x=c_d_race_17["Race/Ethnicity"], y=c_d_race_17["No. Deaths"]), row=1, col=2)
        figure3.update_layout(title_text="Proportions of Cases and Deaths by Race and Ethnicity Among Ages 0‐17", width = 1100)
        st.plotly_chart(figure3)
    if chosen_age_group == "Proportions of Cases and Deaths by Race and Ethnicity Among Ages 18‐34":
        url3 = "https://docs.google.com/spreadsheets/d/1TXocxC8tcETauiLwgYZV_WkHOK7X3e5w1cFpvoK8TyI/edit#gid=0"
        path3 = url3.replace('/edit#gid=', '/export?format=csv&gid=')
        c_d_race_34 = pd.read_csv(path3, encoding="utf8")

        figure4 = make_subplots(rows=1, cols=2)

        figure4.add_trace(
    go.Bar(name='Cases', x=c_d_race_34["Race/Ethnicity"], y=c_d_race_34["No. Cases"]),
    row=1, col=1
)

        figure4.add_trace(
    go.Bar(name='Deaths', x=c_d_race_34["Race/Ethnicity"], y=c_d_race_34["No. Deaths"]),
    row=1, col=2
)

        figure4.update_layout(title_text="Proportions of Cases and Deaths by Race and Ethnicity Among Ages 18-34", width = 1100)
        st.plotly_chart(figure4)

    if chosen_age_group == "Proportions of Cases and Deaths by Race and Ethnicity Among Ages 35‐49":
        url4 = "https://docs.google.com/spreadsheets/d/1_sMMrETw52k9yd4vX3j_wx1-aF29XpIg2rXIgb5sVks/edit#gid=0"
        path4 = url4.replace('/edit#gid=', '/export?format=csv&gid=')
        c_d_race_49 = pd.read_csv(path4, encoding="utf8")

        figure5 = make_subplots(rows=1, cols=2)

        figure5.add_trace(
        go.Bar(name='Cases', x=c_d_race_49["Race/Ethnicity"], y=c_d_race_49["No. Cases"]),
        row=1, col=1
)

        figure5.add_trace(
        go.Bar(name='Deaths', x=c_d_race_49["Race/Ethnicity"], y=c_d_race_49["No. Deaths"]),
        row=1, col=2
)

        figure5.update_layout(title_text="Proportions of Cases and Deaths by Race and Ethnicity Among Ages 35-49", width = 1100)
        st.plotly_chart(figure5)
    
    if chosen_age_group == "Proportions of Cases and Deaths by Race and Ethnicity Among Ages 50‐64":
        url5 = "https://docs.google.com/spreadsheets/d/1mv4gXtQ13hEzVa4W6A2fi3WagAk6hY8Vc8FsaDjYv70/edit#gid=0"
        path5 = url5.replace('/edit#gid=', '/export?format=csv&gid=')
        c_d_race_64 = pd.read_csv(path5, encoding="utf8")
        figure6 = make_subplots(rows=1, cols=2)

        figure6.add_trace(
    go.Bar(name='Cases', x=c_d_race_64["Race/Ethnicity"], y=c_d_race_64["No. Cases"]),
    row=1, col=1
)

        figure6.add_trace(
    go.Bar(name='Deaths', x=c_d_race_64["Race/Ethnicity"], y=c_d_race_64["No. Deaths"]),
    row=1, col=2
)

        figure6.update_layout(title_text="Proportions of Cases and Deaths by Race and Ethnicity Among Ages 50-64", width =1100)
        st.plotly_chart(figure6)

    if chosen_age_group == "Proportions of Cases and Deaths by Race and Ethnicity Among Ages 65-79":
        url6 = "https://docs.google.com/spreadsheets/d/1kQiIBrhMffAV0pcNknmipMocSt1unrJVolwcM59rtc4/edit#gid=0"
        path6 = url6.replace('/edit#gid=', '/export?format=csv&gid=')
        c_d_race_79 = pd.read_csv(path6, encoding="utf8")
        url6 = "https://docs.google.com/spreadsheets/d/1kQiIBrhMffAV0pcNknmipMocSt1unrJVolwcM59rtc4/edit#gid=0"
        path6 = url6.replace('/edit#gid=', '/export?format=csv&gid=')
        c_drace_79 = pd.read_csv(path6,encoding="utf8")
        figure7 = make_subplots(rows=1, cols=2)
        figure7.add_trace(
    go.Bar(name='Cases', x=c_d_race_79["Race/Ethnicity"], y=c_d_race_79["No. Cases"]),
    row=1, col=1
)
        figure7.add_trace(
    go.Bar(name='Deaths', x=c_d_race_79["Race/Ethnicity"], y=c_d_race_79["No. Deaths"]),
    row=1, col=2
)

        figure7.update_layout(title_text="Proportions of Cases and Deaths by Race and Ethnicity Among Ages 65-79", width = 1100)
        st.plotly_chart(figure7)

    if chosen_age_group == "Proportions of Cases and Deaths by Race and Ethnicity Among Ages 80+":
        url7 = "https://docs.google.com/spreadsheets/d/1i-fZFBNUGenfOIgMOMTOprB6_Nm8JjRrLNWRqEGvjcU/edit#gid=0"
        path7 = url7.replace('/edit#gid=', '/export?format=csv&gid=')
        c_d_race_80 = pd.read_csv(path7, encoding="utf8")
        
        figure8 = make_subplots(rows=1, cols=2)
        figure8.add_trace(
    go.Bar(name='Cases', x=c_d_race_80["Race/Ethnicity"], y=c_d_race_80["No. Cases"]),
    row=1, col=1
)

        figure8.add_trace(
    go.Bar(name='Deaths', x=c_d_race_80["Race/Ethnicity"], y=c_d_race_80["No. Deaths"]),
    row=1, col=2
)

        figure8.update_layout(title_text="Proportions of Cases and Deaths by Race and Ethnicity Among Ages 80+", width = 1100)
        st.plotly_chart(figure8)



if menu_id == "Overview":
    st.markdown("""
COVID-Local provides basic key metrics against which to assess pandemic response and progress toward reopening.  

📈 Phase 2: Initial re-opening: Current esetimate of <25 cases per 100,000 population per day  
📉 Phase 3: Economic recovery: Current estimate of <10 cases per 100,000 population per day   

*daily testing data currently available only for Los Angeles County, Orange County, and San Diego County  

for more details related to thresholds please see  
See more at https://www.covidlocal.org/metrics/.    
For additional information please contact *ahm44@aub.edu.lb*.  
""")
    st.markdown(f"""
    One of the key metrics for which data are widely available is the estimate of **daily new cases per 100,000
    population**.

    Here, in following graphics, we will track:

    (A) Estimates of daily new cases per 100,000 population (averaged over the last seven days)  
    
    (B) Daily incidence (new cases)  
    
    (C) Cumulative cases and deaths  
    
    (D) Daily new tests*  

    Data source: Data for cases are procured automatically from **COVID-19 Data Repository by the Center for Systems Science and Engineering (CSSE) at Johns Hopkins University**.  
    
    The data is updated at least once a day or sometimes twice a day in the [COVID-19 Data Repository](https://github.com/CSSEGISandData/COVID-19).  

    Infection rate, positive test rate, ICU headroom and contacts traced from https://covidactnow.org/.  

    *Calculation of % positive tests depends on consistent reporting of county-wise total number of tests performed routinely. Rolling averages and proportions are not calculated if reporting is inconsistent over a period of 14 days.  

    *Report updated on {str(today)}.*  
    """)

