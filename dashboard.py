import geopandas
import streamlit as st 
import pandas as pd 
import numpy as np
import folium 

from streamlit_folium import folium_static
from folium.plugins import MarkerCluster

import plotly.express as px

from datetime import datetime 

# Decorador utilizado para funções. 
# Habilitando "mutation" quer dizer que a variável da função não é fixa!
@st.cache(allow_output_mutation=True)
def get_data(path): 
    data = pd.read_csv(path)
    return data

@st.cache(allow_output_mutation=True)
def get_geofile(url): 
    geofile = geopandas.read_file(url)
    return geofile 

def set_feature(data): 
    # add new features
    data['m2_living'] = data['sqft_living'].apply(lambda x: x * 0.092903)
    data['price_m2'] = data['price'] / data['m2_living']
    return data

def overview_data(data):
    # ==================
    # Data Overview
    # ==================
    f_attributes = st.sidebar.multiselect('Enter columns', data.columns)
    f_zipcode = st.sidebar.multiselect('Enter Zipcode', data['zipcode'].unique())

    st.title('Data Overview')

    # attributes + zipcode = Selecionar colunas e linhas
    # attributes = Selecionar colunas
    # zipcode = Selecionar linhas
    # 0 + 0 = Retorno o dataset original

    if ((f_zipcode != []) & (f_attributes != [])):
        data = data.loc[data['zipcode'].isin(f_zipcode), f_attributes]
    elif ((f_zipcode != []) & (f_attributes == [])):
        data = data.loc[data['zipcode'].isin(f_zipcode), :]
    elif ((f_zipcode == []) & (f_attributes != [])):
        data = data.loc[:, f_attributes]
    else:
        data = data.copy()

    st.dataframe(data.head())

    # Colocando tabelas lado a lado 
    c1, c2 = st.columns((1, 1.6))

    # Average Metrics
    df1 = data[['id', 'zipcode']].groupby('zipcode').count().reset_index()
    df2 = data[['price', 'zipcode']].groupby('zipcode').mean().reset_index()
    df3 = data[['m2_living', 'zipcode']].groupby('zipcode').mean().reset_index()
    df4 = data[['price_m2', 'zipcode']].groupby('zipcode').mean().reset_index()

    # Merge 
    m1 = pd.merge(df1, df2, on='zipcode', how='inner')
    m2 = pd.merge(m1, df3, on='zipcode', how='inner')
    df = pd.merge(m2, df3, on='zipcode', how='inner')

    df.colums = ['zipcode', 'total houses', 'price', 'm2 living', 'price/m2']

    # st.dataframe(df, height=600)
    c1.header('Average Values')
    c1.dataframe(df, height=600)

    # Statistic Descriptive
    num_attributes = data.select_dtypes(include=['int64', 'float64'])
    media = pd.DataFrame(num_attributes.apply(np.mean))
    mediana = pd.DataFrame(num_attributes.apply(np.median))
    std = pd.DataFrame(num_attributes.apply(np.std))

    max = pd.DataFrame(num_attributes.apply(np.max))
    min = pd.DataFrame(num_attributes.apply(np.min))

    df1 = pd.concat([max, min, media, mediana, std], axis=1).reset_index()
    df1.columns = ['attributes', 'max', 'min', 'mean', 'median', 'std']

    # st.dataframe(df1, height=600)
    c2.header('Descriptive Analysis')
    c2.dataframe(df1, height=600)

    return None

def portfolio_density(data, geofile):
    # ======================
    # Portfolio Density
    # ======================

    st.title('Region Overview')

    c1, c2 = st.columns((1, 1))

    c1.header('Portifolio Density')

    df = data.sample(10)

    # Base Map - Folium
    density_map = folium.Map(location=[data['lat'].mean(), data['long'].mean()], default_zoom_start=15)

    marker_cluster = MarkerCluster().add_to(density_map)
    for name, row in df.iterrows():
        folium.Marker([row['lat'], row['long']], 
        popup = 'Sold R${0} on: {1}. Features: {2} sqft, {3} bedrooms, {4} bathrooms, year built: {5}'.format(
            row['price'], row['date'], row['sqft_living'], row['bedrooms'], row['bathrooms'], row['yr_built']
        )).add_to(marker_cluster)

    with c1: 
        folium_static(density_map)

    # Region Price Map
    c2.header('Price Density')
    df = data[['price', 'zipcode']].groupby('zipcode').mean().reset_index()
    df.columns = ['zipcode', 'price']

    url = 'https://opendata.arcgis.com/datasets/83fc2e72903343aabff6de8cb445b81c_2.geojson'
    geofile = get_geofile(url)

    region_price_map = folium.Map(width=600, location=[data['lat'].mean(), data['long'].mean()], default_zoom_strat=15)

    region_price_map.choropleth(data=df, geo_data=geofile, columns=['zipcode', 'price'], 
                                key_on='feature.properties.ZIP', fill_color='YlOrRd', fill_opacity=0.7, 
                                line_opacity=0.2, legend_name='AVG PRICE')
    with c2: 
        folium_static(region_price_map)

    return None

def commercial_destribution(data):
    # ======================================================
    # Distribuição dos imoveis por categorias comerciais
    # ======================================================
    st.sidebar.title('Commercial Options')
    st.title('Commercial Attributes')

    # Average Price per Year

    data['date'] = pd.to_datetime(data['date']).dt.strftime('%Y-%m-%d')

    # filters
    min_year_built = int(data['yr_built'].min())
    max_year_built = int(data['yr_built'].max())

    st.sidebar.subheader('Select Max Year Built')
    f_year_built = st.sidebar.slider('Year Built', min_year_built, max_year_built, min_year_built)

    # Data Selection
    df = data.loc[data['yr_built'] < f_year_built]
    df = df[['yr_built', 'price']].groupby('yr_built').mean().reset_index()

    # Plot
    st.header('Average Price per Year Built')
    fig = px.line(df, x='yr_built', y='price')
    st.plotly_chart(fig, use_container_width=True)

    # Average Price per Day
    st.header('Average Price per Day')
    st.sidebar.subheader('Select Max Date')

    # Filter
    min_date = datetime.strptime(data['date'].min(), '%Y-%m-%d') 
    max_date = datetime.strptime(data['date'].max(), '%Y-%m-%d') 

    f_date = st.sidebar.slider('Date', min_date, max_date, min_date)
    data['date'] = pd.to_datetime(data['date'])

    # Data Selection
    df = data.loc[data['date'] < f_date]
    df = df[['date', 'price']].groupby('date').mean().reset_index()

    # Plot
    fig = px.line(df, x='date', y='price')
    st.plotly_chart(fig, use_container_width=True)


    # -------------- Histograma
    st.header('Price Distribution')
    st.sidebar.subheader('Select Max Price')

    # Filter 
    min_price = int(data['price'].min())
    max_price = int(data['price'].max())
    avg_price = int(data['price'].mean())

    # Data Selection
    f_price = st.sidebar.slider('Price', min_price, max_price, avg_price)
    df = data.loc[data['price'] < f_price]

    # Plot
    fig = px.histogram(df, x='price', nbins=50)
    st.plotly_chart(fig, use_container_width=True)

    return None

def attributes_destribution(data): 
    #====================================================
    # Distribuição dos imoveis por categorias fisicas
    #====================================================
    st.sidebar.title('Attributes Options')
    st.sidebar.subheader('House Attributes')

    # Filters
    f_bedrooms = st.sidebar.selectbox('Max number of bedrooms', sorted(set(data['bedrooms'].unique())))
    f_bathrooms = st.sidebar.selectbox('Max number of bathrooms', sorted(set(data['bathrooms'].unique())))

    c1, c2 = st.columns(2)

    # House per bedrooms
    c1.header('Houses per bedrooms')
    df = data.loc[data['bedrooms'] < f_bedrooms, 'bedrooms']
    fig = px.histogram(df, x='bedrooms', nbins=19)
    c1.plotly_chart(fig, use_container_width=True)

    # House per bathrooms
    c2.header('Houses per bathrooms')
    df = data.loc[data['bathrooms'] < f_bathrooms, 'bathrooms']
    fig = px.histogram(df, x='bathrooms', nbins=19)
    c2.plotly_chart(fig, use_container_width=True)

    # Filters
    f_floors = st.sidebar.selectbox('Max number of floor', sorted(set(data['floors'].unique())))
    f_waterview = st.sidebar.checkbox('Only Houses with Water View')

    c1, c2 = st.columns(2)

    # House per floors
    c1.header('Houses per Floor')
    df = data.loc[data['floors'] < f_floors, 'floors']
    fig = px.histogram(data, x='floors', nbins=19)
    c1.plotly_chart(fig, use_container_width=True)

    # House per water_view
    if f_waterview: 
        df = data.loc[data['waterfront'] == 1]
    else: 
        df = data.copy()

    c2.header('Houses WaterView')
    fig = px.histogram(df, x='waterfront', nbins=10)
    c2.plotly_chart(fig, use_container_width=True)

    return None

if __name__ == "__main__":

    st.set_page_config(layout='wide')

    # Escrevendo um título na página 
    st.title('House Rocket')

    # ETL
    #Data Extraction
    path = 'kc_house_data.csv'
    url = 'https://opendata.arcgis.com/datasets/83fc2e72903343aabff6de8cb445b81c_2.geojson'

    data = get_data(path)
    geofile = get_geofile(url)

    #Transformation
    data = set_feature(data)

    overview_data(data)
    
    portfolio_density(data, geofile)

    commercial_destribution(data)

    attributes_destribution(data)