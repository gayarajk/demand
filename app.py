import os
import datetime
import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow
import seaborn as sns
import streamlit as st
from fpdf import FPDF
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARIMA
import pmdarima as pm
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
import base64
from tensorflow.keras.models import Sequential
from keras.layers import Dense, LSTM
#from fbprophet import Prophet
# from fbprophet.diagnostics import performance_metrics
# from fbprophet.diagnostics import cross_validation
# from fbprophet.plot import plot_cross_validation_metric
# from sklearn.decomposition import PCA
#from sklearn.model_selection import train_test_split
import io
from PIL import Image


#####Do display GCU logo
#image=Image.open("GCU_logo.PNG")
#st.image(image, use_column_width=True)

###Title Page
st.title('Paychex Demand Driven Resource Allocation Analysis')

###Stop messages for plots
st.set_option('deprecation.showPyplotGlobalUse', False)


 ####To upload data files to User Interface
def get_dataset(option_name):   
   upload_file=st.file_uploader("Choose a csv file", type='csv')
   data=pd.read_csv(upload_file)
   st.write(data.head(10))
   st.success("Loaded sucessfully")
   
 ####Download link for report  
def create_download_link(val, filename):
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}.pdf">Download file</a>'

def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(look_back, len(dataset)):
        a = dataset[i-look_back:i, 0]
        X.append(a)
        Y.append(dataset[i, 0])
    return np.array(X), np.array(Y)
   
           
data = pd.read_csv("f_pay_data.csv")   
service = data.serviceName.unique()
option_name=st.sidebar.selectbox('Select option',('Upload Dataset','EDA','Visualization', 'Model', 'Help')) 

#metric = data.metricName.unique()   
if option_name == 'Upload Dataset':    
   d = get_dataset(option_name)
   fdisptype=st.radio("Select display format?", ('CSV', 'Tab'))
   if fdisptype == 'CSV':
      csv_data = data.head(10)
      csv_data = csv_data.to_csv()
      st.write(csv_data)
   elif fdisptype=='Tab':
      tab_data = data.head(10)
      tab_data = tab_data.to_csv( sep ='\t') 
      st.write(tab_data)
   ####Radio button to select to explore data after uploading	  
   fcoltype=st.radio("About Data?", ('Information','Size', 'Columns', 'Rows'))
   x = data.describe()
   
   if fcoltype == 'Information':
      st.write(x) 
   elif fcoltype == 'Size':    
       st.write(data.count())
   elif fcoltype == 'Columns':
        st.write(data.dtypes) 
   elif fcoltype == 'Rows':
        st.write(data.shape)   
     # data.info(buf=buffer)
      #s = buffer.getvalue()
      #with open("df_info.txt", "w",encoding="utf-8") as f:
       #    f.write(s)
          
    ####EDA selection    
elif option_name == 'EDA':
   service_name=st.sidebar.selectbox('Select service name',(service))
   rslt_df = data[data['serviceName'] == service_name] 
   st.write(rslt_df.head(10))
   fcoltype=st.radio("About Data?", ('Information','Size', 'Columns', 'Rows', 'Correlation', 'Data with NaN' , 'Data with null'))
   x = rslt_df.describe()
   ####### EDA radio button selection for EDA to get detail information about data
   if fcoltype == 'Information':
      st.write(x) 
   elif fcoltype == 'Size':    
       st.write(rslt_df.count())
   elif fcoltype == 'Columns':
        st.write(rslt_df.dtypes) 
   elif fcoltype == 'Rows':
        st.write(rslt_df.shape)
   elif fcoltype == 'Correlation':
        st.write(rslt_df.corr()) 
   elif fcoltype == 'Data with NaN':
        st.write(rslt_df[rslt_df.isna().any(axis=1)])   
   elif fcoltype == 'Data with null':
        rslt_df[rslt_df.isnull().any(axis=1)]   
   metric = rslt_df.columns.tolist()    
   metric=metric[2:11]
   
   ####Option to select metric name from dropdown in the EDA page
   metric_name=st.sidebar.selectbox('Select metric name',(metric))
   rs = rslt_df[["date", metric_name]]
   st.write(rs)
   date_time = pd.to_datetime(rslt_df.pop('date'), format= '%m/%d/%Y %H:%M')
   plot_cols = ['Average Response Time (ms)', 'Calls per Minute', 'Number of Slow Calls']
   #plot_features = rslt_df[plot_cols]
   #plot_features.index = date_time
   #st.pyplot(plot_features.plot(subplots=True))
   rs.hist()
   st.pyplot()
   ####Daat visulaization page
elif option_name == 'Visualization': 
    st.subheader("Data Visualization")
    data=st.file_uploader("Upload dataset:",type=['csv', 'xlsx','txt','json'])
    st.success("Data successfully loaded")
    
    if data is not None:
       print('test1')
       df=pd.read_csv(data)
       st.dataframe(df.head(50))
       st.success("Data successfully loaded")
       if st.checkbox("Select Multiple columns to plot"):
          selected_columns=st.multiselect('Select your preferred columns', df.columns)
          df1 = df[selected_columns]
          st.dataframe(df)
       if st.checkbox("Display Heatmap"):
          selected_columns=st.multiselect('Select your preferred columns', df.columns)
          df1 = df[selected_columns]
          st.write(sns.heatmap(df1.corr(),vmax=1,square=True,annot=True,cmap='viridis') )
          st.pyplot()
       if st.checkbox("Display Pairplot"):
          st.set_option('deprecation.showPyplotGlobalUse', False)
          selected_columns=st.multiselect('Select your preferred columns', df.columns)
          df1 = df[selected_columns]
          st.write(sns.pairplot(df1, diag_kind='kde'))
          st.pyplot()
		  
		 ####Select Model from the list
       
elif option_name == 'Model':
    modelname=st.sidebar.selectbox('Select model',( 'ARIMA','LSTM')) 
    #st.subheader("Model")
	#####Time series analysis
    if modelname == 'Time series':
       data = pd.read_csv("f_pay_data.csv")   
       service = data.serviceName.unique()
       metric_name = data.columns
       service_name=st.sidebar.selectbox('Select service name',(service))
       metric=st.sidebar.selectbox('Select metric', data.columns)
       data=st.file_uploader("Upload dataset:",type=['csv', 'xlsx','txt','json'])
       if data is not None:
          print('test1')
          data=pd.read_csv(data)
          rslt_df = data[data['serviceName'] == service_name]
          st.write(rslt_df.head(50))
          st.success("Data successfully loaded")

          date_time = pd.to_datetime(rslt_df.pop('date')) 
          plot_cols = ['Average Response Time (ms)', 'Calls per Minute', 'Number of Slow Calls', 'Stall Count', 'Errors per Minute', 'HTTP Error Codes per Minute' ]
          plot_features = rslt_df[plot_cols]
          plot_features.index = date_time
          st.write(plot_features.plot(subplots=True))
          st.pyplot()  
          col_feautures = rslt_df[metric]
          report_text = st.text_input("Report Text")
          export_as_pdf = st.button("Export Report")
          if export_as_pdf:
             pdf = FPDF()
             pdf.add_page()
             pdf.set_font('Arial', 'B', 16)
             pdf.cell(40, 10, st.pyplot())
    
             html = create_download_link(pdf.output(dest="S").encode("latin-1"), "test")

             st.markdown(html, unsafe_allow_html=True)  
			 ###########LTSM
    elif modelname == 'LSTM':
       st.subheader("LSTM Model") 
       data = pd.read_csv("f_pay_data.csv")   
       service = data.serviceName.unique()
       metric_name = data.columns
       service_name=st.sidebar.selectbox('Select service name',(service))
       metric=st.sidebar.selectbox('Select metric', data.columns)
       data=st.file_uploader("Upload dataset:",type=['csv', 'xlsx','txt','json'])
       if data is not None:
          print('test1')
          data=pd.read_csv(data)
          rslt_df = data[data['serviceName'] == service_name]
          st.write(rslt_df.head(50))
          st.success("Data successfully loaded") 
          
          #scaled_data = scaler.fit_transform(dataset)
          #scaled_data[:10]
          #look_back = 52
          
          train_size = int(0.85 * len(rslt_df))
          test_size = len(rslt_df) - train_size

          univariate_df = rslt_df[['date', metric ]].copy()
          univariate_df.columns = ['ds', 'y']

          #train = univariate_df.iloc[:train_size, :]
		  

          #x_train, y_train = pd.DataFrame(univariate_df.iloc[:train_size, 0]), pd.DataFrame(univariate_df.iloc[:train_size, 1])
          #x_valid, y_valid = pd.DataFrame(univariate_df.iloc[train_size:, 0]), pd.DataFrame(univariate_df.iloc[train_size:, 1])
          data = univariate_df.filter(['y'])
          #Convert the dataframe to a numpy array
          dataset = data.values

          scaler = MinMaxScaler(feature_range=(-1, 0))
          scaled_data = scaler.fit_transform(dataset)

          scaled_data[:10]

          #st.write('Train:',len(train), 'Test:',len(x_valid))
          # Defines the rolling window
          look_back = 52
          # Split into train and test sets
		  ###########################Split data into train and testing
          train, test = scaled_data[:train_size-look_back,:], scaled_data[train_size-look_back:,:]



          x_train, y_train = create_dataset(train, look_back)
          x_test, y_test = create_dataset(test, look_back)

          # reshape input to be [samples, time steps, features]
          x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
          x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))

          st.write(len(x_train), len(x_test))
          #Build the LSTM model
          model = Sequential()
          model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
          model.add(LSTM(64, return_sequences=False))
          model.add(Dense(25))
          model.add(Dense(1))
		  
		  #####building the model

          # Compile the model
          model.compile(optimizer='adam', loss='mean_squared_error')

          #Train the model
          model.fit(x_train, y_train, batch_size=1, epochs=5, validation_data=(x_test, y_test))

          st.write(model.summary())
          # Lets predict with the model
          train_predict = model.predict(x_train)
          test_predict = model.predict(x_test)

          # invert predictions
          train_predict = scaler.inverse_transform(train_predict)
          y_train = scaler.inverse_transform([y_train])

          test_predict = scaler.inverse_transform(test_predict)
          y_test = scaler.inverse_transform([y_test])

          ###### Get the root mean squared error (RMSE) and MAE
          score_rmse = np.sqrt(mean_squared_error(y_test[0], test_predict[:,0]))
          score_mae = mean_absolute_error(y_test[0], test_predict[:,0])
          st.write('RMSE: {}'.format(score_rmse))
          x_train_ticks = univariate_df.head(train_size)['ds']
          y_train = univariate_df.head(train_size)['y']
          x_test_ticks = univariate_df.tail(test_size)['ds']

          ######Plot the forecast
          f, ax = plt.subplots(1)
          f.set_figheight(6)
          f.set_figwidth(15)

          sns.lineplot(x=x_train_ticks, y=y_train, ax=ax, label='Train Set') #navajowhite
          sns.lineplot(x=x_test_ticks, y=test_predict[:,0], ax=ax, color='green', label='Prediction') #navajowhite
          sns.lineplot(x=x_test_ticks, y=y_test[0], ax=ax, color='orange', label='Ground truth') #navajowhite

          ax.set_title(f'Prediction \n MAE: {score_mae:.2f}, RMSE: {score_rmse:.2f}', fontsize=14)
          ax.set_xlabel(xlabel='Date', fontsize=14)
          ax.set_ylabel(ylabel=metric, fontsize=14)

          st.pyplot()
          
    elif modelname == 'ARIMA':
       st.subheader("ARIMA MODEL")
       data = pd.read_csv("f_pay_data.csv")   
       service = data.serviceName.unique()
       metric_name = data.columns
       service_name=st.sidebar.selectbox('Select service name',(service))
       metric=st.sidebar.selectbox('Select metric', data.columns)
       data=st.file_uploader("Upload dataset:",type=['csv', 'xlsx','txt','json'])
       if data is not None:
          print('test1')
          data=pd.read_csv(data)
          rslt_df = data[data['serviceName'] == service_name]
          st.write(rslt_df.head(50))
          st.success("Data successfully loaded") 
          data1 = data.filter(['metric'])
          train_size = int(0.85 * len(rslt_df))
          test_size = len(rslt_df) - train_size

          univariate_df = rslt_df[['date', metric ]].copy()
          univariate_df.columns = ['ds', 'y']

          train = univariate_df.iloc[:train_size, :]
###########################Split data into train and testing
          x_train, y_train = pd.DataFrame(univariate_df.iloc[:train_size, 0]), pd.DataFrame(univariate_df.iloc[:train_size, 1])
          x_valid, y_valid = pd.DataFrame(univariate_df.iloc[train_size:, 0]), pd.DataFrame(univariate_df.iloc[train_size:, 1])

          st.write('Train:',len(train), 'Test:',len(x_valid))
          # Fit model
          model = ARIMA(y_train, order=(1,1,1))
          model_fit = model.fit()

          # Prediction with ARIMA
          y_pred, se, conf = model_fit.forecast(len(x_valid))

           # Calcuate metrics
          score_mae = mean_absolute_error(y_valid, y_pred)
          score_rmse = math.sqrt(mean_squared_error(y_valid, y_pred))

          st.write('RMSE: {}'.format(score_rmse))
          f, ax = plt.subplots(1)
          f.set_figheight(4)
          f.set_figwidth(15)

          model_fit.plot_predict(1, 100, ax=ax)
          sns.lineplot(x=x_valid.index, y=y_valid['y'], ax=ax, color='orange', label='Ground truth') #navajowhite

          ax.set_title(f'Prediction \n MAE: {score_mae:.2f}, RMSE: {score_rmse:.2f}', fontsize=14)
          ax.set_xlabel(xlabel='Date', fontsize=14)
          ax.set_ylabel(ylabel=metric, fontsize=14)
          tdf = rslt_df[metric]
          mn = tdf.min(axis=0)
          mx = tdf.max(axis=0)

          ax.set_ylim(mn, mx)
          st.pyplot()
          
          f, ax = plt.subplots(1)
          f.set_figheight(4)
          f.set_figwidth(15)

          sns.lineplot(x=x_valid.index, y=y_pred, ax=ax, color='blue', label='predicted') #navajowhite
          sns.lineplot(x=x_valid.index, y=y_valid['y'], ax=ax, color='orange', label='Ground truth') #navajowhite

          ax.set_xlabel(xlabel='Date', fontsize=14)
          ax.set_ylabel(ylabel=metric, fontsize=14)

          st.pyplot()
          
          st.text("---"*100)
          st.text("---"*100)
          
          ###Auto Arima
          

          model = pm.auto_arima(y_train, start_p=1, start_q=1, 
          test='adf',       # use adftest to find optimal 'd'
          max_p=3, max_q=3, # maximum p and q
          m=1,              # frequency of series
          d=None,           # let model determine 'd'
          seasonal=False,   # No Seasonality
          start_P=0, 
          D=0, 
          trace=True,
          error_action='ignore',  
          suppress_warnings=True, 
          stepwise=True)

          st.write(model.summary())
          model.plot_diagnostics(figsize=(16,8))
          st.pyplot()
                
                
          
   ##########Help Page       
elif option_name == "Help":
     st.subheader("Help Page")
     st.write('This is an interactive webpage for Machine learning project. You can upload the dataset and select the page you want to navigate.\
     Data file name: f_pay_data.csv')
          
 

