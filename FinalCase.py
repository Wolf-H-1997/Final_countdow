#!/usr/bin/env python
# coding: utf-8

# In[24]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import requests
import json
import folium
from folium import plugins
from folium import IFrame
from streamlit_folium import folium_static
import plotly.express as px
import streamlit as st
import plotly.graph_objects as go
import sklearn
from plotly.subplots import make_subplots
from sklearn.metrics import r2_score



# ## Importing Data

# In[25]:


PaxNL=pd.read_csv("Pax movement clean.csv") #"C:\\Users\\charl\\Downloads\\Pax movement clean.csv"
PaxNLyear=pd.read_csv("full_years_PaxNL.csv") #C:\\Users\\charl\\Downloads\\full_years_PaxNL.csv"
PaxNLmonths=pd.read_csv("PaxNL_months.csv") #"C:\\Users\\charl\\Downloads\\PaxNLmonths.csv"
PaxNLkwartaal=pd.read_csv("PaxNL_kwartaal.csv")
PaxSchiphol=pd.read_csv("83435NED_TypedDataSet_06112023_144215.csv", sep=";",index_col=0)
PaxLanden=pd.read_csv("83435NED_metadata.csv",index_col=0)


# In[3]:


LandenSchiphol = pd.merge(PaxSchiphol, PaxLanden, on='LuchthavensHerkomstBestemming')


# In[4]:


# Rename columns
LandenSchiphol = LandenSchiphol.rename(columns={'TotaalAantalPassagiers_1': 'Totaal Passagiers', 'PassagiersAangekomenInNederland_2': 'Passagiers Aangekomen','PassagiersVertrokkenUitNederland_3' : 'Passagiers Vertrokken'})

# Rearrange columns
LandenSchiphol = LandenSchiphol[['Land','Perioden', 'Totaal Passagiers', 'Passagiers Aangekomen','Passagiers Vertrokken']]  # Replace with the actual column names in the desired order


# In[5]:


LandenSchiphol.isna().sum()


# In[6]:


# Drop rows with NaN values
LandenSchiphol.dropna(inplace=True)


# In[7]:


# Extracting the year from the 'Column' column
LandenSchiphol['Perioden'] = LandenSchiphol['Perioden'].str[:4]


# In[8]:


LandenSchiphol.info()


# In[9]:


LandenSchiphol['Perioden'] = pd.to_datetime(LandenSchiphol['Perioden'])
LandenSchiphol['Perioden'] = LandenSchiphol['Perioden'].dt.year


# In[10]:



# Define a custom mapping for the Dutch month names
dutch_to_english_month = {
    'januari': 'January',
    'februari': 'February',
    'maart': 'March',
    'april': 'April',
    'mei': 'May',
    'juni': 'June',
    'juli': 'July',
    'augustus': 'August',
    'september': 'September',
    'oktober': 'October',
    'november': 'November',
    'december': 'December'
}
# Define the function to convert to year and month
def convert_to_datetime(period):
    if period.isdigit() and len(period) == 4:
        return period
    elif 'kwartaal' in period:
        year = period.split(' ')[0]
        quarter = period.split(' ')[1]
        quarter_mapping = {'1e': '04', '2e': '07', '3e': '10', '4e': '12'}
        return year + '-' + quarter_mapping[quarter]
    elif any(month in period for month in dutch_to_english_month.keys()):
        for key in dutch_to_english_month.keys():
            if key in period:
                period = period.replace(key, dutch_to_english_month[key])
                break
        return pd.to_datetime(period, format='%Y %B').strftime('%Y-%m')
    else:
        return period

# Apply the function to the 'Perioden' column
PaxNL['Perioden'] = PaxNL['Perioden'].apply(convert_to_datetime)


# In[11]:


headers = {
    'accept': 'application/json',
    'resourceversion': 'v4',
    'app_id': '97e1b46b',  
    'app_key': '7bd51b9a4763b4f63b523c8a4bf827e2'  
}


api_url = 'https://api.schiphol.nl/public-flights/flights' 


response = requests.get(api_url, headers=headers)

# Check if the request was successful
if response.status_code == 200:
    # The API request was successful, and the response data is in response.text
    data = response.json()  # If the response is in JSON format
    
    # Now, you can normalize the JSON data into a Pandas DataFrame
    df = pd.json_normalize(data)
else:
    # The API request was not successful, and you may want to handle errors here
    print(f"API request failed with status code: {response.status_code}")


# In[12]:


flight_df = pd.DataFrame(data)
df4 = pd.json_normalize(flight_df.flights)


# In[13]:


ipa4 = df4[['scheduleTime', 'scheduleDate', 'flightDirection', 'flightName','route.destinations' , 'lastUpdatedAt','prefixICAO', 'aircraftType.iataMain', 'gate', 'terminal']] 


# In[14]:


PaxNLmonths['Perioden'] = pd.to_datetime(PaxNLmonths['Perioden'], format='%d/%m/%Y')
PaxNLyear['Perioden'] = pd.to_datetime(PaxNLyear['Perioden'], format='%Y')
PaxNLyear['Jaar'] = PaxNLyear['Perioden'].dt.year


# ## 1D inspecties

# #### Jaarlijks

# In[15]:


PaxNLyear['Jaar'] = PaxNLyear['Perioden'].dt.year


# In[16]:


st.title('Bar plot van de totale vluchten per jaar')

fig = px.bar(PaxNLyear, x='Jaar', y='Totaal Vluchten', color='Luchthavens', barmode='group',
             title='Totale vluchten per jaar',
             color_discrete_sequence=px.colors.qualitative.Plotly)

st.plotly_chart(fig)


# In[17]:



excluded_value = "Totaal luchthavens van nationaal belang"
years = PaxNLyear['Jaar'].unique()
colors = ['rgb(93, 164, 214)', 'rgb(255, 144, 14)', 'rgb(44, 160, 101)', 'rgb(255, 65, 54)', 'rgb(207, 114, 255)']

# Create subplots with dropdown menus
fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'domain'}]])

buttons = []
visible_list = [True] + [False] * (len(years) - 1)

for i, year in enumerate(years):
    data_year = PaxNLyear[(PaxNLyear['Jaar'] == year) & (PaxNLyear['Luchthavens'] != excluded_value)]
    fig.add_trace(go.Pie(values=data_year['Aantal passagiers'], labels=data_year['Luchthavens'], name=f"Year {year}",
                         marker_colors=colors, visible=visible_list[i]), row=1, col=1)
    
    button = dict(
        method="update",
        label=f"Year {year}",
        args=[{"visible": [False] * i + [True] + [False] * (len(years) - i - 1)}, {}]
    )
    buttons.append(button)

updatemenu = list([
    dict(active=0,
         buttons=list([
            dict(label='Aantal passagiers',
                 method='update',
                 args=[{'visible': [True, False]}, {'title': 'Aantal passagiers per jaar'}]),
            dict(label='Aantal cargo(ton)',
                 method='update',
                 args=[{'visible': [False, True]}, {'title': 'Aantal cargo(ton) per jaar'}])
            ]),
    )
])

year_buttons = []
for i, year in enumerate(years):
    year_button = dict(
        method="update",
        label=str(year),
        args=[{"visible": [True if j == i else False for j in range(len(years))]}, {}]
    )
    year_buttons.append(year_button)

updatemenu.append(dict(buttons=year_buttons, direction="down", showactive=True, x=0.1, xanchor="left", y=1.1, yanchor="top"))

fig.update_layout(title_text="Aantal passagiers per jaar", updatemenus=updatemenu)

st.plotly_chart(fig)


# #### Per kwartaal

# ***a bar chart per year to show the amount of flights/passengers/cargo per kwartaal

# ### Geospatial Data

# In[18]:



PaxNL_Amsterdam = PaxNLyear[PaxNLyear.Luchthavens == 'Amsterdam Airport Schiphol']
PaxNL_Rotterdam = PaxNLyear[PaxNLyear.Luchthavens == 'Rotterdam The Hague Airport']
PaxNL_Eindhoven = PaxNLyear[PaxNLyear.Luchthavens == 'Eindhoven Airport']
PaxNL_Maastricht = PaxNLyear[PaxNLyear.Luchthavens == 'Maastricht Aachen Airport']
PaxNL_Groningen = PaxNLyear[PaxNLyear.Luchthavens == 'Groningen Airport Eelde']


# In[19]:


airports = {
    'Amsterdam Airport Schiphol': [52.308056, 4.764167],
    'Rotterdam The Hague Airport': [51.9555086, 4.4398832],
    'Eindhoven Airport': [51.4583691, 5.3920556],
    'Maastricht Aachen Airport': [50.91249905, 5.77132050283004],
    'Groningen Airport Eelde': [53.1214959, 6.58172323449291]
}
filtered_df = PaxNLyear[PaxNLyear['Luchthavens'].isin(airports.keys())]


# In[20]:


schiphol = [52.308056, 4.764167]
rotterdam = [51.9555086, 4.4398832]
eindhoven = [51.4583691, 5.3920556]
maastricht = [50.91249905, 5.77132050283004]
groningen = [53.1214959, 6.58172323449291]


# ### Scatterplot met Statische analyse

# In[21]:

# Assuming you have already loaded PaxNLmonths DataFrame

# Define colors
colors = {
    'Totaal luchthavens van nationaal belang': 'darkblue',
    'Amsterdam Airport Schiphol': 'gold',
    'Rotterdam The Hague Airport': 'limegreen',
    'Eindhoven Airport': 'red',
    'Maastricht Aachen Airport': 'purple',
    'Groningen Airport Eelde': 'yellow'
}

# Create a scatter plot for each airport
fig = go.Figure()

# Initialize lists to store data for the table
airports = []
residuals_data = []
r2_data = []

for airport, color in colors.items():
    df = PaxNLmonths[PaxNLmonths['Luchthavens'] == airport]
    fig.add_trace(go.Scatter(x=df['Perioden'], y=df['Totaal Vluchten'], mode='markers', name=airport, marker=dict(color=color)))

    # Compute the trendline for each airport
    x = pd.to_numeric(df['Perioden'])
    y = np.log(df['Totaal Vluchten'])

    # Check for infinite or too large values
    mask = np.isfinite(y)
    x = x[mask]
    y = y[mask]

    slope, intercept = np.polyfit(x, y, 1)

    # Store the data in the lists
    airports.append(airport)
    residuals_data.append(df['Totaal Vluchten'] - np.exp(intercept + slope * x))
    r2_data.append(r2_score(y, intercept + slope * x))

    # Create a DataFrame for the forecast for each airport
    forecast_dates = pd.date_range(start='2022-01-01', end='2025-01-01', freq='M')
    forecast_df = pd.DataFrame({'Perioden': forecast_dates})
    forecast_df['Totaal Vluchten'] = np.exp(intercept + slope * pd.to_numeric(forecast_df['Perioden']))

    # Add the forecast line for each airport
    fig.add_trace(go.Scatter(x=forecast_df['Perioden'], y=forecast_df['Totaal Vluchten'], mode='lines', name=f'{airport} Forecast', line=dict(color=color)))

# Update the layout
fig.update_layout(
    title='Vluchten per jaar',
    xaxis_title='Tijd',
    yaxis_title='Totaal Vluchten (log)',
    showlegend=True,
    yaxis_type="log",
    xaxis=dict(range=["2022-01-01", "2025-01-01"])  # Extend the range until 2027
)

# Create a DataFrame for the table
table_df = pd.DataFrame({
    'Airport': airports,
    'Residuals': residuals_data,
    'R-squared': r2_data
})

# Display the table
st.dataframe(table_df)
st.plotly_chart(fig)


# ## Hulpbronnen

# In[22]:



# Group the data by 'Land' and 'Perioden' and sum the 'Totaal Passagiers' values
summed_data = LandenSchiphol.groupby(['Land', 'Perioden'])['Totaal Passagiers'].sum().reset_index()

# Sort the data for each year separately
sorted_data_2019 = LandenSchiphol[LandenSchiphol['Perioden'] == 2019].groupby('Land')[['Passagiers Aangekomen', 'Passagiers Vertrokken']].sum().sort_values('Passagiers Aangekomen', ascending=False)
sorted_data_2020 = LandenSchiphol[LandenSchiphol['Perioden'] == 2020].groupby('Land')[['Passagiers Aangekomen', 'Passagiers Vertrokken']].sum().sort_values('Passagiers Aangekomen', ascending=False)
sorted_data_2021 = LandenSchiphol[LandenSchiphol['Perioden'] == 2021].groupby('Land')[['Passagiers Aangekomen', 'Passagiers Vertrokken']].sum().sort_values('Passagiers Aangekomen', ascending=False)
sorted_data_2022 = LandenSchiphol[LandenSchiphol['Perioden'] == 2022].groupby('Land')[['Passagiers Aangekomen', 'Passagiers Vertrokken']].sum().sort_values('Passagiers Aangekomen', ascending=False)

st.title("Passenger Details by Country")

# Create figure
fig = go.Figure()

# Add traces for each year
for data, year in zip([sorted_data_2019, sorted_data_2020, sorted_data_2021, sorted_data_2022], [2019, 2020, 2021, 2022]):
    fig.add_trace(
        go.Bar(x=data.index, y=data['Passagiers Aangekomen'], name=f'Passagiers Aangekomen - {year}', marker_color='orange')
    )
    fig.add_trace(
        go.Bar(x=data.index, y=data['Passagiers Vertrokken'], name=f'Passagiers Vertrokken - {year}', marker_color='blue')
    )

# Set title
fig.update_layout(
    title_text="Passenger Details by Country",
    xaxis_title="Country",
    yaxis_title="Passenger Count",
)

# Add dropdown for the year
fig.update_layout(
    updatemenus=[
        dict(
            buttons=list([
                dict(label="All",
                     method="update",
                     args=[{"visible": [True, True, True, True, True, True, True, True]},
                           {"title": "All"}]),
                dict(label="2019",
                     method="update",
                     args=[{"visible": [True, True, False, False, False, False, False, False]},
                           {"title": "2019"}]),
                dict(label="2020",
                     method="update",
                     args=[{"visible": [False, False, True, True, False, False, False, False]},
                           {"title": "2020"}]),
                dict(label="2021",
                     method="update",
                     args=[{"visible": [False, False, False, False, True, True, False, False]},
                           {"title": "2021"}]),
                dict(label="2022",
                     method="update",
                     args=[{"visible": [False, False, False, False, False, False, True, True]},
                           {"title": "2022"}]),
            ]),
        )
    ]
)

# Add range slider
fig.update_layout(
    xaxis=dict(
        type="category",
        rangeslider=dict(
            visible=True
        )
    ),
    barmode='stack'  # Add this line to stack the bars
)

st.plotly_chart(fig)


# # Streamlit

# In[23]:


map_center = [52.1326, 5.2913]

# maak  kaart
kaart = folium.Map(location=map_center, zoom_start=7, max_bounds=True)

# zwart wit
folium.TileLayer('cartodbpositron', name='CartoDB Positron').add_to(kaart)

# vliegveld coördinaten
schiphol = [52.308056, 4.764167]
rotterdam = [51.9555086, 4.4398832]
eindhoven = [51.4583691, 5.3920556]
maastricht = [50.91249905, 5.77132050283004]
groningen = [53.1214959, 6.58172323449291]


# Maak apparte popups
schiphol_popup = folium.Popup("Schiphol Airport", parse_html=True)
rotterdam_popup = folium.Popup("Rotterdam The Hague Airport")
eindhoven_popup = folium.Popup("Eindhoven Airport")
maastricht_popup = folium.Popup("Maastricht Aachen Airport")
groningen_popup = folium.Popup("Groningen Airport Eelde")
# HTML content voor in de pop ups
popup_content_schiphol = f"""
<b>Flight Information at Schiphol Airport:</b><br>
<table>
    <tr>
        <th>Flight Name</th>
        <th>scheduleDate</th>
        <th>Schedule Time</th>
        <th>route destinations</th>

    </tr>
"""
popup_content_rotterdam = f"""
<b>Flight Information at Rotterdam The Hague Airport:</b><br>
<table>
    <tr>
        <th>Perioden</th>
        <th>Totaal aantal vluchten</th>
        <th>Aantal passagiers</th>
    </tr>
"""
popup_content_eindhoven = f"""
<b>Flight Information at Eindhoven Airport:</b><br>
<table>
    <tr>
        <th>Perioden</th>
        <th>Totaal aantal vluchten</th>
        <th>Aantal passagiers</th>
    </tr>
"""
popup_content_maastricht = f"""
<b>Flight Information at Maastricht Aachen Airport:</b><br>
<table>
    <tr>
        <th>Perioden</th>
        <th>Totaal aantal vluchten</th>
        <th>Aantal passagiers</th>
    </tr>
"""
popup_content_groningen = f"""
<b>Flight Information at Groningen Airport Eelde:</b><br>
<table>
    <tr>
        <th>Perioden</th>
        <th>Totaal aantal vluchten</th>
        <th>Aantal passagiers</th>
    </tr>
"""

for _, row in PaxNL_Rotterdam.iterrows():
    popup_content_rotterdam += f"""
    <tr>
        <td>{row['Jaar']}</td>
        <td>{row['Totaal Vluchten']}</td>
        <td>{row['Aantal passagiers']}</td>
    <tr>
    """
for _, row in PaxNL_Eindhoven.iterrows():
    popup_content_eindhoven += f"""
    <tr>
        <td>{row['Jaar']}</td>
        <td>{row['Totaal Vluchten']}</td>
        <td>{row['Aantal passagiers']}</td>
    <tr>
    """
for _, row in PaxNL_Maastricht.iterrows():
    popup_content_maastricht += f"""
    <tr>
        <td>{row['Jaar']}</td>
        <td>{row['Totaal Vluchten']}</td>
        <td>{row['Aantal passagiers']}</td>
    <tr>
    """
for _, row in PaxNL_Groningen.iterrows():
    popup_content_groningen += f"""
    <tr>
        <td>{row['Jaar']}</td>
        <td>{row['Totaal Vluchten']}</td>
        <td>{row['Aantal passagiers']}</td>
    <tr>
    """

# Schiphol met de informatie van de api
for _, row in df4.iterrows():
    popup_content_schiphol += f"""
    <tr>
        <td>{row['flightName']}</td>
        <td>{row['scheduleDate']}</td>
        <td>{row['scheduleTime']}</td>
        <td>{row['route.destinations']}</td>
    </tr>
"""

# Close the HTML table and popup content
#popup_content_schiphol += "</table>"
#popup_content_rotterdam += "</table>"

# voeg de popups toe aan de kaart
folium.Marker(schiphol, popup=popup_content_schiphol).add_to(kaart)
folium.Marker(rotterdam, popup=popup_content_rotterdam).add_to(kaart)
folium.Marker(eindhoven, popup=popup_content_eindhoven).add_to(kaart)
folium.Marker(maastricht, popup=popup_content_maastricht).add_to(kaart)
folium.Marker(groningen, popup=popup_content_groningen).add_to(kaart)



# laat de kaart zien
st.write(kaart)

