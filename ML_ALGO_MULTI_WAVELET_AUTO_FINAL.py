#while rerunning the code make sure all the ML Model folders are renamed or deleted because it will overwrite the exsisitng !!!!
from sklearn.multioutput import MultiOutputRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from numpy import mean
from numpy import std
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle
import glob
import joblib
import warnings
import os
import threading
import logging
import time
import math
from math import sqrt
import datetime
import warnings
from numpy import array
from keras.models import Sequential,Model,load_model,save_model
from keras.layers import Activation,Dropout, Dense
from keras.layers import Flatten,LSTM
from keras.layers import GlobalMaxPooling1D
from keras.models import Model
from keras.layers import Embedding
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow import keras
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import glob
import sys
from keras.layers import SimpleRNN
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from multiprocessing import Process
from sklearn import preprocessing
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
logging.basicConfig(filename="ML_Process_Log.txt", level=logging.DEBUG, format="%(asctime)s %(message)s", filemode="w")
import psutil
from tqdm import tqdm
from time import sleep
import psutil
import logging
import pywt
import lightgbm as lgb
from lightgbm import LGBMRegressor
import xgboost as xgb
from xgboost import XGBRegressor

cwd=os.getcwd()
st_all= time.time()
class preprocess:
  
  def specs(self,name):
    global cwd_mem
    with open(cwd_mem+"usage.txt", "a") as f:
      f.write("\n")
      f.write(name)
      f.write("\n")
      f.write("########################")
      f.write("\n")
      cpu_start=psutil.cpu_percent()
      f.write('Total CPUs utilized percentage:')
      f.write(str(cpu_start))
      f.write("\n")
      per=psutil.virtual_memory().percent
      f.write('Total RAM utilized percentage:')
      f.write(str(per))
      f.write("\n")
      f.write("Available memory (Gb):")
      mem_aval=psutil.virtual_memory()[1]
      mem_aval=mem_aval/1024/1024/1024
      f.write(str(mem_aval))
      f.write("\n")
      f.write("##########################")
 
  def dwt_train(self):
    os.chdir(cwd+"\\train\\")
    files=cwd+"\\train\\"
    file_path=files
    os.makedirs(cwd+"\\train_wav\\",exist_ok=True)
    t1=glob.glob(file_path+"*.csv")
    new_data=pd.DataFrame()
    global wavelet_level
    global wavelet_mode
    global wavelet_type
    print("Wavelet preprocessing!!!!")
    wavelet_type = input("enter the wavelet type :")
    wavelet_mode = input("enter the wavelet mode :")
    wavelet_level= int(input("enter the level of wavlet :"))
    print("Converting time domain data into wavelet coefficients of Train files !!!!!")   
    for file in t1:
        cols=[]
        filename=file.replace(file_path,"") #read file from path into induvidual files
        filename1= filename.replace(".csv","")
        df=pd.read_csv(file)
        del df['Sim_time']
        proc=df['process']
        volt=df['voltage']
        temp=df['temperature']
        coeffs_dwt=pd.DataFrame()
        cols=df.columns
        cols= [e for e in cols if e not in ('process', 'voltage','temperature')]
        for col in cols:
          coeffs=pywt.wavedec(df[col],wavelet_type,mode=wavelet_mode,level=wavelet_level)
          slice=pywt.coeffs_to_array(coeffs)[0]
          coeffs_dwt[col]=slice
        coeffs_dwt['process']=proc
        coeffs_dwt['voltage']=volt
        coeffs_dwt['temperature']=temp
        #coeffs_dwt = coeffs_dwt.dropna()
        coeffs_dwt = coeffs_dwt.fillna(method='ffill')
        coeffs_dwt.to_csv(cwd+"\\train_wav\\"+filename1+".csv",index=False)
    print("Wavelet convertion successful of Train files!!!!!")

  def dwt_test(self):
    os.chdir(cwd+"\\test\\")
    files=cwd+"\\test\\"
    file_path=files
    os.makedirs(cwd+"/test_wav/",exist_ok=True)
    t1=glob.glob(file_path+"*.csv")
    new_data=pd.DataFrame()
    print("Converting time domain data into wavelet coefficients of Test files!!!!!") 
    for file in t1:
        cols=[]
        filename=file.replace(file_path,"") #read file from path into induvidual files
        filename1= filename.replace(".csv","")
        df=pd.read_csv(file)
        proc=df['process']
        volt=df['voltage']
        temp=df['temperature']
        time=df['Sim_time']
        del df['Sim_time']
        cols=df.columns
        cols= [e for e in cols if e not in ('process', 'voltage','temperature')]
        coeffs_dwt=pd.DataFrame()
        for col in cols:
          coeffs=pywt.wavedec(df[col],wavelet_type,mode=wavelet_mode,level=wavelet_level)
          slice=pywt.coeffs_to_array(coeffs)[0]
          coeffs_dwt[col]=slice
        coeffs_dwt['Sim_time']=time
        coeffs_dwt['process']=proc
        coeffs_dwt['voltage']=volt
        coeffs_dwt['temperature']=temp
        coeffs_dwt = coeffs_dwt.fillna(method='ffill')
        coeffs_dwt.to_csv(cwd+"\\test_wav\\"+filename1+".csv",index=False) 
    print("Wavelet convertion successful for Test files!!!!!")

  def combine_files(self):
    global combined_csv
    st = time.time()
    stime = datetime.datetime.now()
    print("Combining csv start time:-", stime)
    print("Data Processing started ")
    logging.debug("Data Processing......")
    os.chdir(cwd+"/train_wav/")# folder path    
    CHUNK_SIZE = 5000
    extension = 'csv'
    csv_file_list= [i for i in glob.glob('*.{}'.format(extension))]
    combined_csv=cwd+"/combined_train_wavelet.csv"
    first_one = True
    for csv_file_name in csv_file_list:
      if not first_one: # if it is not the first csv file then skip the header row (row 0) of that file
       skip_row = [0]
      else:
       skip_row = []
      chunk_container = pd.read_csv(csv_file_name, chunksize=CHUNK_SIZE, skiprows = skip_row,header=None)
      for chunk in chunk_container:
       chunk.to_csv(combined_csv, mode="a",index=False,header=False)
      first_one = False    
    et = time.time()
    etime = datetime.datetime.now()
    print("combining csv end time:-", etime)
    logging.debug("Data processing completed")
    elapsed_time = et - st
    print('Execution time for combining csv is:', elapsed_time, 'seconds')
    print('Data Processing completed')

  def fetch(self):
    global combined_csv
    print("Fetching Data.....")
    st = time.time()
    stime = datetime.datetime.now()
    print("data fetch start time:-", stime)
    combined_csv=pd.read_csv(cwd+"/"+"combined_train_wavelet.csv",chunksize=5000) # split into chunks rather than storing into memory all at once
    combined_csv=pd.concat(combined_csv)
    #combined_csv['process'] = combined_csv['process'].replace({'Nominal':1,'weak':2,'strong':3,'nominal':1})
    #label_encoder = preprocessing.LabelEncoder()
    #combined_csv['process']= label_encoder.fit_transform(combined_csv['process'])
    #print(combined_csv['process'])
    etime = datetime.datetime.now()
    print("data fetch end time:-", etime)
    print('Data Fetched Sucessfully.....')
    et = time.time()
    elapsed_time = et - st
    print('Execution time for fetching data is:', elapsed_time, 'seconds')
  
class RandomForest:
  global cwd_rf
  global wavelet_level
  global wavelet_mode
  global wavelet_type
  os.makedirs(cwd+"/RandomForest/",exist_ok=True)
  cwd_rf=cwd+"/RandomForest/"
  #model creation using trained data
  def model_creation(self):
    logging.debug("Model Creation started")
    print("Random Forest Model Creation Started.....")
    st = time.time()
    stime = datetime.datetime.now()
    print("Random Forest Model creation start time:-", stime)
    os.makedirs(cwd_rf+"/models",exist_ok=True)
    my_file = open(cwd+"/input_signals.txt", "r")
    lines = [line.strip() for line in my_file]
    #reading output signals
    my_file1 = open(cwd+"/output_signals.txt", "r")
    lines1= [line.strip() for line in my_file1] 
    inputs=lines
    outputs=lines1
    X=combined_csv[inputs].values
    est=int(input("enter the no of trees for random forest modelling:"))
    #leaf=int(input("enter the min_sample_leaf of trees for random forest modelling:"))
    #min_split=int(input("enter the min_sample_split for random forest modelling:"))
    for i in outputs:
      print("Creating model for signal-",i)
      y=combined_csv[i].values
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
      #rf= RandomForestRegressor(n_estimators =est,n_jobs=-1,min_samples_split=min_split,min_samples_leaf=leaf)# use all cores available for modelling
      rf= RandomForestRegressor(n_estimators =est,n_jobs=-1)
      print("model fitting")
      model=rf.fit(X_train,y_train)
      print("fitting completed")
      y_train_predicted = model.predict(X_train)
      y_test_predicted = model.predict(X_test)
      mse_train = mean_squared_error(y_train, y_train_predicted)
      mse_test = mean_squared_error(y_test, y_test_predicted)
      print("Random forest, Train MSE: {} Test MSE: {}".format(mse_train, mse_test))
      os.makedirs(cwd_rf+"/models",exist_ok=True)
      model_path=cwd_rf+"/models/signal_"+i+".joblib"
      joblib.dump(model,open(model_path,'wb'),compress=3)
    et = time.time()
    etime = datetime.datetime.now()
    print("Model creation end time:-", etime)
    elapsed_time = et - st
    print('Execution time for creating model is :', elapsed_time, 'seconds')
    logging.debug("model saved sucessfully")
    print("RANDOM FOREST MODEL SAVED SUCESSFULLY")
    
  #prediction of test file using trained model
  def prediction(self):
    st = time.time()
    print("RANDOM FOREST PREDICTION STARTED...")
    logging.debug("prediction started")
    stime = datetime.datetime.now()
    print("Prediction start time:-", stime)
    file_path=cwd+"\\test_wav\\"
    t1=glob.glob(file_path+"*.csv")
    for file in t1:
      filename=file.replace(file_path,"") #read file from path into induvidual files
      filename1= filename.replace(".csv","")
      df=pd.read_csv(file)
      logging.debug("signals from input file are read sucessfully for prediction")
      my_file = open(cwd+"/input_signals.txt", "r")
      # reading the file
      lines= [line.strip() for line in my_file] 
      my_file1 = open(cwd+"/output_signals.txt", "r")
      # reading the file
      lines1= [line.strip() for line in my_file1] 
      logging.debug("signals from output file are read sucessfully for prediction")
      inputs=lines
      outputs=lines1
      for i in outputs:
       row=df[inputs].values
       model_path=cwd_rf+"/models/signal_"+i+".joblib"
       model=model_path
       regression=joblib.load(open(model,'rb'))
       logging.debug("trained model loaded sucessfully")
       os.makedirs(cwd_rf+"/predictions/"+i,exist_ok=True)
       yhat =regression.predict(row)
       name="output"+i
       name=pd.DataFrame()
       name['Sim_time']=df['Sim_time']
       name['predicted_signal']=yhat#acess numpy array colum wise
       name['actual_signal']=df[i]
       name.to_csv(cwd_rf+"/predictions/"+i+"/"+filename1+".csv")
    et = time.time()  
    etime = datetime.datetime.now()
    print("Prediction end time:-", etime)
    elapsed_time = et - st
    print('Execution time for prediction is :', elapsed_time, 'seconds')
    logging.debug("prediction file saved sucessfully")
    print("PREDICTION COMPLETED SUCESSFULLY")
    
  def idwt(self):
    global wavelet_level
    global wavelet_mode
    global wavelet_type
    global time
    st=time.time()
    logging.debug("IDWT CONVERSION STARTED-RF")
    print("IDWT CONVERSION STARTED [RF] ...")
    def get_csv_files(folder):
      csv_files = []
      for file in os.listdir(folder):
          if file.endswith('.csv'):
              csv_files.append(file)
      return csv_files
    my_file1 = open(cwd+"/output_signals.txt", "r")
    # reading the file
    lines1= [line.strip() for line in my_file1] 
    wavelet_type = input("enter the wavelet type :")
    wavelet_mode = input("enter the wavelet mode :")
    wavelet_level= int(input("enter the level of wavlet :"))
    for i in lines1:
      folder1_path = cwd_rf+"predictions/"+i+"/"
      folder2_path = cwd+"/test/"
      new=pd.DataFrame()
      os.makedirs(cwd_rf+"predictions_idwt/"+i,exist_ok=True)
      files1 = get_csv_files(folder1_path)
      files2 = get_csv_files(folder2_path)
      for file1 in files1:
        if file1 in files2:
            file1_path = os.path.join(folder1_path, file1)
            file2_path = os.path.join(folder2_path, file1)
            with open(file1_path, 'r') as f1, open(file2_path, 'r') as f2:
                pred= pd.read_csv(f1)
                Time= pd.read_csv(f2)
                actual=Time[i]
                coeffs=pred['predicted_signal']
                sim_time=Time['Sim_time']
                sz =len(Time)
                dummy_signal = [i for i in range(0, sz)]
                coeffs_dummy = pywt.wavedec(dummy_signal,wavelet_type,wavelet_mode,wavelet_level)
                coeff_slices = pywt.coeffs_to_array(coeffs_dummy)[1]
                coeffs_idwt=pywt.array_to_coeffs(coeffs,coeff_slices,output_format='wavedec')
                coeffs_idwt1=pywt.waverec(coeffs_idwt,wavelet_type,wavelet_mode)
                new=pd.DataFrame()
                coeffs_idwt1=coeffs_idwt1[0:sz]
                new['actual_signal']=actual
                new['predicted_signal']=coeffs_idwt1
                new['Sim_time']=sim_time
                new.to_csv(cwd_rf+'predictions_idwt/'+i+"/"+file1)
      et = time.time()
      elapsed_time = et - st
      print('Execution time for IDWT conversion is :', elapsed_time, 'seconds')
      logging.debug("IDWT conversion sucessfull")
      print("WAVELET TO TIME DOMAIN CONVERSION COMPLETED SUCESSFULLY !!!!")
  

  #plotting graph between actual signal and predicetd signal  
  # def graph_plot(self):
  #   st = time.time()
  #   cwd_rf=cwd+"/RandomForest/"
  #   print("Generating Graph Plot")
  #   my_file1 = open(cwd+"/output_signals.txt", "r")
  #   lines1= [line.strip() for line in my_file1] 
  #   for i in lines1:
  #     path = cwd_rf+"/predictions_idwt/"+i+"/"+"*.csv"
  #     lpath = len(path)-5
  #     list_files = []
  #     count =0
  #     os.makedirs(cwd_rf+"/graphs/"+i,exist_ok=True)
  #     for file in glob.glob(path):
  #      count = count+1
  #      name = file[lpath:]
  #      name = name[:-4]
  #      list_files += [name]
  #      df_fun=pd.read_csv(file)
  #      X_time=df_fun['Sim_time']
  #      vinn = df_fun['actual_signal']
  #      pred_vinn = df_fun['predicted_signal']
  #      fig=plt.figure(figsize=(16,9),facecolor='w', edgecolor='k')
  #      plt.plot(X_time,vinn,color="blue", linewidth=3, label = vinn)
  #      plt.plot(X_time,pred_vinn, color="red", linewidth=3, label = pred_vinn)
  #      title = name
  #      plt.xlabel("Time", fontsize=10)
  #      plt.ylabel("Pll_out", fontsize=10)
  #      plt.grid(True)
  #      plt.legend()
  #      plt.legend(["Actual signal","Predicted signal"], loc ="lower right")
  #      plt.title(title)
  #      nm =cwd_rf+"/graphs/"+i+"/"+name+".png"
  #      plt.savefig(nm)
  #     et = time.time()
  #   elapsed_time = et - st
  #   print('Execution time for graph plot is :', elapsed_time, 'seconds')
  #   logging.debug("graph plotted sucessfull")
  #   print("GRAPH PLOTTED SUCESSFULLY")
  
  def graph_plot(self):
    st = time.time()
    cwd_rf = cwd + "/RandomForest/"
    print("Generating Graph Plot with Metrics")
    
    my_file1 = open(cwd + "/output_signals.txt", "r")
    lines1 = [line.strip() for line in my_file1]
    
    for i in lines1:
        path = cwd_rf + "/predictions_idwt/" + i + "/" + "*.csv"
        lpath = len(path) - 5
        os.makedirs(cwd_rf + "/graphs/" + i, exist_ok=True)
        
        for file in glob.glob(path):
            name = file[lpath:-4]
            df_fun = pd.read_csv(file)
            X_time = df_fun['Sim_time']
            actual = df_fun['actual_signal']
            predicted = df_fun['predicted_signal']
            
            # Metrics Calculation
            df_fun["diff"] = actual - predicted
            n = len(df_fun) - 1
            upper = actual.pow(2).sum() / n
            lower = df_fun['diff'].pow(2).sum() / n
            snr = 10 * math.log10(upper / lower) if lower > 0 else float('inf')
            
            mae = mean_absolute_error(actual, predicted)
            mse = mean_squared_error(actual, predicted)
            rmse = sqrt(mse)
            r2 = r2_score(predicted,actual)
            
            # Plot the graph
            fig, ax = plt.subplots(figsize=(16, 9), facecolor='w', edgecolor='k')
            ax.plot(X_time, actual, color="blue", linewidth=2, label="Actual Signal")
            ax.plot(X_time, predicted, color="red", linewidth=2, label="Predicted Signal")
            
            # Add metrics as text on the graph
            metrics_text = f"MAE: {mae:.4f}\nMSE: {mse:.4f}\nRMSE: {rmse:.4f}\nR²: {r2:.4f}\nSNR: {snr:.4f} dB"
            ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, fontsize=12,
                    verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='lightgray'))
            
            ax.set_title(name, fontsize=14)
            ax.set_xlabel("Time", fontsize=12)
            ax.set_ylabel("Pll_out", fontsize=12)
            ax.grid(True)
            ax.legend(loc="lower right")
            
            # Save the graph
            nm = cwd_rf + "/graphs/" + i + "/" + name + ".png"
            plt.savefig(nm)
            plt.close()
    
    et = time.time()
    elapsed_time = et - st
    print('Execution time for graph plot is:', elapsed_time, 'seconds')
    logging.debug("Graph plotted successfully")
    print("GRAPH PLOTTED SUCCESSFULLY")

  #calculating metrics using prediction file
  def metrics(self):
    st=time.time()
    logging.debug("metrics calculation started")
    print("METRICS CALCULATION STARTED...")
    my_file1 = open(cwd+"/output_signals.txt", "r")
    lines1= [line.strip() for line in my_file1] 
    for i in lines1:
      file_path = cwd_rf+"/predictions_idwt/"+i+"/"
      t1=glob.glob(file_path+"*.csv")
      result = pd.DataFrame(columns = ['FILENAME','MAE','MSE','RMSE','R2SCORE','SNR'])
      os.makedirs(cwd_rf+"/metrics/"+i,exist_ok=True)
      for file in t1:
       filename=file.replace(file_path,"") #read file from path into induvidual files
       filename1= filename.replace(".csv","")
       df=pd.read_csv(file)
       df["diff"] = df["actual_signal"] - df["predicted_signal"]
       length_of_rows=len(df)
       n=length_of_rows-1
       upper=0
       lower=0
       upper = df['actual_signal'].pow(2).sum()
       lower = df['diff'].pow(2).sum()
       upper = upper/n
       lower = lower/n
       snr=10*math.log10(upper/lower)
       predvinn=df['predicted_signal']
       vinn=df['actual_signal']
       rmse=sqrt(mean_squared_error(predvinn,vinn))
       mae = mean_absolute_error(predvinn,vinn)
       mse = mean_squared_error(predvinn,vinn)
       r2=r2_score(predvinn,vinn)
       #result= result.append({'FILENAME':filename1,'MAE':mae,'MSE':mse,'RMSE':rmse,'R2SCORE':r2,'SNR':snr},ignore_index=True)
       new_row = pd.DataFrame([{'FILENAME': filename1, 'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2SCORE': r2, 'SNR': snr}])
       result = pd.concat([result, new_row], ignore_index=True)
       result_file =cwd_rf+"/metrics/"+i+"/"+"metrics"+".csv"
       result.to_csv(result_file,index=False)
    et= time.time()
    elapsed_time = et - st
    print('Execution time for metrics calculation is :', elapsed_time, 'seconds')
    logging.debug("metrics calculation completed sucessfully")
    print("METRICS CALCULATION COMPLETED SUCESSFULLY")

#class MLP
class MLP:
#model creation using trained data
  def model_creation(self):
    st = time.time()
    global cwd_mlp
    os.makedirs(cwd+"/mlp/",exist_ok=True)
    cwd_mlp=cwd+"/mlp/"
    print("MLP Model Creation started")
    logging.debug("files in train folder combined for modeling")
    combined_csv=pd.read_csv(cwd+"/combined_train_wavelet.csv",chunksize=5000000)
    combined_csv=pd.concat(combined_csv)
    my_file = open(cwd+"/input_signals.txt", "r")
    #reading file
    lines = [line.strip() for line in my_file]
    logging.debug("signals from input file are read sucessfully")
    #reading output signals
    my_file1 = open(cwd+"/output_signals.txt", "r")
    # reading the file
    lines1 = [line.strip() for line in my_file1]
    logging.debug("signals from output file are read sucessfully")
    inputs=lines
    X=combined_csv[inputs].values
    outputs=lines1
    for i in outputs:
      os.makedirs(cwd_mlp+"/models/"+i,exist_ok=True)
      y=combined_csv[i].values
      logging.debug("input and output values are assigned to X and y")
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
      logging.debug("input,output data has been split for train and test")
      logging.debug("model creation started")
      model = Sequential()
      model.add(Dense(50,activation='relu'))
      model.add(Dense(30,activation='relu'))#hidden layer
      model.add(Dense(1))#output layer
      # model.compile(optimizer=Adam(lr=0.001), loss='mse')
      model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
      history="history"+i
      history=model.fit(x=X_train,y=y_train,validation_data=(X_test,y_test),batch_size=32,epochs=30)
      logging.debug("model created sucessfully")
      # model.save(cwd_mlp+"/models/"+i+"/")
      model.save(cwd_mlp + "/models/" + i + "/" + "model.keras")
      # summarize history for accuracy and loss
      plt.plot(history.history['loss'])
      plt.plot(history.history['val_loss'])
      plt.title('model loss')
      plt.ylabel('loss')
      plt.xlabel('epoch')
      plt.legend(['train', 'val'], loc='upper left')
      plt.savefig(cwd_mlp+"/models/"+i+"/"+"model_graph_"+i+".png")
      et = time.time()
      elapsed_time = et - st
      print('Execution time for creating model is :', elapsed_time, 'seconds')
      logging.debug("model saved sucessfully")
      print("MLP MODEL SAVED SUCESSFULLY")
#prediction of test file using trained model
  def prediction(self):
    st = time.time()
    cwd_mlp=cwd+"/mlp/"
    print("MLP PREDICTION STARTED...")
    logging.debug("prediction started")
    file_path=cwd+"\\test_wav\\"
    t1=glob.glob(file_path+"*.csv")
    output=pd.DataFrame()
    for file in t1:
      filename=file.replace(file_path,"") #read file from path into induvidual files
      filename1= filename.replace(".csv","")
      df=pd.read_csv(file)
      logging.debug("signals from input file are read sucessfully for prediction")
      my_file = open(cwd+"/input_signals.txt", "r")
      # reading the file
      lines = [line.strip() for line in my_file]
      my_file1 = open(cwd+"/output_signals.txt", "r")
      # reading the file
      lines1 = [line.strip() for line in my_file1]
      logging.debug("signals from output file are read sucessfully for prediction")
      outputs=lines1
      for i in outputs:
       my_model = keras.models.load_model(cwd_mlp+"/models/"+i+"/model.keras")
       os.makedirs(cwd_mlp+"/predictions/"+i+"/",exist_ok=True)
       row=df[lines].values
       logging.debug("trained model loaded sucessfully")
       prediction=my_model.predict(row,verbose=0)
       prediction = [item for sublist in prediction for item in sublist]
       name="output"+i
       name=pd.DataFrame()
       name['Sim_time']=df['Sim_time']
       name['predicted_signal']=prediction#acess numpy array colum wise
       name['actual_signal']=df[i]
       name.to_csv(cwd_mlp+"/predictions/"+i+"/"+filename1+".csv")
    et = time.time()
    elapsed_time = et - st
    print('Execution time for prediction is :', elapsed_time, 'seconds')
    logging.debug("prediction file saved sucessfully")
    print("PREDICTION COMPLETED SUCESSFULLY")
  
  def idwt(self):
    global time
    st=time.time()
    cwd_mlp=cwd+"/mlp/"
    logging.debug("IDWT CONVERSION STARTED")
    print("IDWT CONVERSION STARTED [MLP]...")
    def get_csv_files(folder):
      csv_files = []
      for file in os.listdir(folder):
          if file.endswith('.csv'):
              csv_files.append(file)
      return csv_files
    my_file1 = open(cwd+"/output_signals.txt", "r")
    # reading the file
    lines1= [line.strip() for line in my_file1] 
    for i in lines1:
      folder1_path = cwd_mlp+"predictions/"+i+"/"
      print(folder1_path)
      folder2_path = cwd+"/test/"
      print(folder2_path)
      new=pd.DataFrame()
      os.makedirs(cwd_mlp+"predictions_idwt/"+i,exist_ok=True)
      files1 = get_csv_files(folder1_path)
      files2 = get_csv_files(folder2_path)
      wavelet_type = input("enter the wavelet type :")
      wavelet_mode = input("enter the wavelet mode :")
      wavelet_level= int(input("enter the level of wavlet :"))
      for file1 in files1:
        if file1 in files2:
            file1_path = os.path.join(folder1_path, file1)
            file2_path = os.path.join(folder2_path, file1)
            with open(file1_path, 'r') as f1, open(file2_path, 'r') as f2:
                pred= pd.read_csv(f1)
                Time= pd.read_csv(f2)
                actual=Time[i]
                coeffs=pred['predicted_signal']
                sim_time=Time['Sim_time']
                sz =len(Time)
                dummy_signal = [i for i in range(0, sz)]
                coeffs_dummy = pywt.wavedec(dummy_signal,wavelet_type,wavelet_mode,wavelet_level)
                coeff_slices = pywt.coeffs_to_array(coeffs_dummy)[1]
                coeffs_idwt=pywt.array_to_coeffs(coeffs,coeff_slices,output_format='wavedec')
                coeffs_idwt1=pywt.waverec(coeffs_idwt,wavelet_type,wavelet_mode)
                new=pd.DataFrame()
                coeffs_idwt1=coeffs_idwt1[0:sz]
                new['actual_signal']=actual
                new['predicted_signal']=coeffs_idwt1
                new['Sim_time']=sim_time
                new.to_csv(cwd_mlp+'predictions_idwt/'+i+'/'+file1)
      et = time.time()
      elapsed_time = et - st
      print('Execution time for IDWT conversion is :', elapsed_time, 'seconds')
      logging.debug("IDWT conversion sucessfull")
      print("WAVELET TO TIME DOMAIN CONVERSION COMPLETED SUCESSFULLY !!!!")

  #plotting graph between actual signal and predicetd signal  
  # def graph_plot(self):
  #   st = time.time()
  #   cwd_mlp=cwd+"/mlp/"
  #   print("Generating Graph Plot")
  #   my_file1 = open(cwd+"/output_signals.txt", "r")
  #   lines1 = [line.strip() for line in my_file1]
  #   for i in lines1:
  #     path = cwd_mlp+"/predictions_idwt/"+i+"/"+"*.csv"
  #     lpath = len(path)-5
  #     list_files = []
  #     count =0
  #     os.makedirs(cwd_mlp+"/graphs/"+i,exist_ok=True)
  #     for file in glob.glob(path):
  #      count = count+1
  #      name = file[lpath:]
  #      name = name[:-4]
  #      list_files += [name]
  #      df_fun=pd.read_csv(file)
  #      X_time=df_fun['Sim_time']
  #      vinn = df_fun['actual_signal']
  #      pred_vinn = df_fun['predicted_signal']
  #      fig=plt.figure(figsize=(16,9),facecolor='w', edgecolor='k')
  #      plt.plot(X_time,vinn,color="blue", linewidth=3, label = vinn)
  #      plt.plot(X_time,pred_vinn, color="red", linewidth=3, label = pred_vinn)
  #      title = name
  #      plt.xlabel("Time", fontsize=10)
  #      plt.ylabel("Qup", fontsize=10)
  #      plt.grid(True)
  #      plt.legend()
  #      plt.legend(["Actual signal","Predicted signal"], loc ="lower right")
  #      plt.title(title)
  #      nm =cwd_mlp+"/graphs/"+i+"/"+name+".png"
  #      plt.savefig(nm)
  #     et = time.time()
  #   elapsed_time = et - st
  #   print('Execution time for graph plot is :', elapsed_time, 'seconds')
  #   logging.debug("graph plotted sucessfull")
  #   print("GRAPH PLOTTED SUCESSFULLY")
  def graph_plot(self):
      st = time.time()
      cwd_mlp = cwd + "/mlp/"
      print("Generating Graph Plot with Metrics")
      
      my_file1 = open(cwd + "/output_signals.txt", "r")
      lines1 = [line.strip() for line in my_file1]
      
      for i in lines1:
          path = cwd_mlp + "/predictions_idwt/" + i + "/" + "*.csv"
          lpath = len(path) - 5
          os.makedirs(cwd_mlp + "/graphs/" + i, exist_ok=True)
          
          for file in glob.glob(path):
              name = file[lpath:-4]
              df_fun = pd.read_csv(file)
              X_time = df_fun['Sim_time']
              actual = df_fun['actual_signal']
              predicted = df_fun['predicted_signal']

              # Metrics Calculation
              df_fun["diff"] = actual - predicted
              n = len(df_fun) - 1
              upper = actual.pow(2).sum() / n
              lower = df_fun['diff'].pow(2).sum() / n
              snr = 10 * math.log10(upper / lower) if lower > 0 else float('inf')

              mae = mean_absolute_error(actual, predicted)
              mse = mean_squared_error(actual, predicted)
              rmse = sqrt(mse)
              r2 = r2_score(actual, predicted)

              # Plotting
              fig, ax = plt.subplots(figsize=(16, 9), facecolor='w', edgecolor='k')
              ax.plot(X_time, actual, color="blue", linewidth=2, label="Actual Signal")
              ax.plot(X_time, predicted, color="red", linewidth=2, label="Predicted Signal")

              # Add metrics as text box
              metrics_text = f"MAE: {mae:.4f}\nMSE: {mse:.4f}\nRMSE: {rmse:.4f}\nR²: {r2:.4f}\nSNR: {snr:.2f} dB"
              ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, fontsize=12,
                      verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='lightgray'))

              ax.set_xlabel("Time", fontsize=12)
              ax.set_ylabel("Qup", fontsize=12)
              ax.grid(True)
              ax.legend(loc="lower right")
              ax.set_title(name)

              nm = cwd_mlp + "/graphs/" + i + "/" + name + ".png"
              plt.savefig(nm)
              plt.close()

      et = time.time()
      elapsed_time = et - st
      print('Execution time for graph plot is:', elapsed_time, 'seconds')
      logging.debug("Graph plotted successfully")
      print("GRAPH PLOTTED SUCCESSFULLY")


  #calculating metrics using prediction file
  def metrics(self):
    st=time.time()
    cwd_mlp=cwd+"/mlp/"
    logging.debug("metrics calculation started")
    print("METRICS CALCULATION STARTED...")
    my_file1 = open(cwd+"/output_signals.txt", "r")
    lines1 = [line.strip() for line in my_file1]
    for i in lines1:
      file_path = cwd_mlp+"/predictions_idwt/"+i+"/"
      t1=glob.glob(file_path+"*.csv")
      result = pd.DataFrame(columns = ['FILENAME','MAE','MSE','RMSE','R2SCORE','SNR'])
      os.makedirs(cwd_mlp+"/metrics/"+i,exist_ok=True)
      for file in t1:
       filename=file.replace(file_path,"") #read file from path into induvidual files
       filename1= filename.replace(".csv","")
       df=pd.read_csv(file)
       df["diff"] = df["actual_signal"] - df["predicted_signal"]
       length_of_rows=len(df)
       n=length_of_rows-1
       upper=0
       lower=0
       upper = df['actual_signal'].pow(2).sum()
       lower = df['diff'].pow(2).sum()
       upper = upper/n
       lower = lower/n
       snr=10*math.log10(upper/lower)
       predvinn=df['predicted_signal']
       vinn=df['actual_signal']
       rmse=sqrt(mean_squared_error(predvinn,vinn))
       mae = mean_absolute_error(predvinn,vinn)
       mse = mean_squared_error(predvinn,vinn)
       r2=r2_score(predvinn,vinn)
       #result= result.append({'FILENAME':filename1,'MAE':mae,'MSE':mse,'RMSE':rmse,'R2SCORE':r2,'SNR':snr},ignore_index=True)
       new_row = pd.DataFrame([{
      'FILENAME': filename1,
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2SCORE': r2,
        'SNR': snr
        }])
       result = pd.concat([result, new_row], ignore_index=True)
       result_file =cwd_mlp+"/metrics/"+i+"/"+"metrics"+".csv"
       result.to_csv(result_file,index=False)
    et= time.time()
    elapsed_time = et - st
    print('Execution time for metrics calculation is :', elapsed_time, 'seconds')
    logging.debug("metrics calculation completed sucessfully")
    print("METRICS CALCULATION COMPLETED SUCESSFULLY")

class RNN:
  global wavelet_level
  global wavelet_type
  global wavelet_mode
  def model_creation(self):
    st = time.time()
    global cwd_rnn
    os.makedirs(cwd+"/lstm/",exist_ok=True)
    cwd_rnn=cwd+"/rnn/"
    print("RNN Model creation Started")
    os.makedirs(cwd_rnn+"/models",exist_ok=True)
    logging.debug("files in train folder combined for modeling")
    combined_csv=pd.read_csv(cwd+"/combined_train_wavelet.csv",chunksize=50000)
    combined_csv=pd.concat(combined_csv)
    my_file = open(cwd+"/input_signals.txt", "r")
    #reading file
    lines = [line.strip() for line in my_file]
    logging.debug("signals from input file are read sucessfully")
    #reading output signals
    my_file1 = open(cwd+"/output_signals.txt", "r")
    # reading the file
    lines1 = [line.strip() for line in my_file1]
    logging.debug("signals from output file are read sucessfully")
    inputs=lines
    X=combined_csv[inputs].values
    X= np.asarray(X)
    outputs=lines1
    for i in outputs:
      os.makedirs(cwd_rnn+"/models/"+i+"/",exist_ok=True)
      y=combined_csv[i].values
      y= np.asarray(y)
      length=len(combined_csv)
      in_len=len(inputs)
      X = X.reshape(length,1,in_len)
      y = array(y)
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
      model = Sequential()
      model.add(SimpleRNN(32,activation='relu',return_sequences=True, input_shape = (1,in_len)))
      model.add(SimpleRNN(16,activation='relu',return_sequences=True))
      model.add(Dense(1))
      model.compile(optimizer=Adam(lr=0.001), loss='mse')
      #early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, restore_best_weights=True)
      history=model.fit(x=X_train,y=y_train,validation_data=(X_test,y_test),batch_size=32,epochs=20)
      plt.plot(history.history['loss'])
      plt.plot(history.history['val_loss'])
      plt.title('model loss')
      plt.ylabel('loss')
      plt.xlabel('epoch')
      plt.legend(['train', 'val'], loc='upper left')
      plt.savefig(cwd_rnn+"/models/"+i+"/"+"model_graph_"+i+".png")
      model.save(cwd_rnn+"/models/"+i+"/")
      et = time.time()
      elapsed_time = et - st
      print('Execution time for creating model is :', elapsed_time, 'seconds')
      logging.debug("model saved sucessfully")
      print("LSTM MODEL SAVED SUCESSFULLY")

  def prediction(self):
    st = time.time()
    cwd_rnn=cwd+"/rnn/"
    print("RNN PREDICTION STARTED...")
    logging.debug("prediction started")
    file_path=cwd+"/test_wav/"
    t1=glob.glob(file_path+"*.csv")
    output=pd.DataFrame()
    for file in t1:
      filename=file.replace(file_path,"") #read file from path into induvidual files
      filename1= filename.replace(".csv","")
      df=pd.read_csv(file)
      logging.debug("signals from input file are read sucessfully for prediction")
      my_file = open(cwd+"/input_signals.txt", "r")
      # reading the file
      lines = [line.strip() for line in my_file]
      my_file1 = open(cwd+"/output_signals.txt", "r")
      # reading the file
      lines1 = [line.strip() for line in my_file1]
      logging.debug("signals from output file are read sucessfully for prediction")
      outputs=lines1
      inputs=lines
      length=len(df)
      in_len=len(inputs)
      for i in outputs:
       my_model = keras.models.load_model(cwd_rnn+"/models/"+i+"/")
       os.makedirs(cwd_rnn+"/predictions/"+i+"/",exist_ok=True)
       row=df[inputs].values
       row = row.reshape(length,1,in_len)
       logging.debug("trained model loaded sucessfully")
       prediction=my_model.predict(row,verbose=0)
       prediction = [item for sublist in prediction for item in sublist]
       name="output"+i
       name=pd.DataFrame()
       name['Sim_time']=df['Sim_time']
       name['predicted_signal']=prediction#acess numpy array colum wise
       name['actual_signal']=df[i]
       name['predicted_signal'] = name['predicted_signal'].apply(lambda x: float(x[0][0]) if isinstance(x, (list, np.ndarray)) and isinstance(x[0], (list, np.ndarray)) else float(x))
       name.to_csv(cwd_rnn+"/predictions/"+i+"/"+filename1+".csv")
    et = time.time()
    elapsed_time = et - st
    print('Execution time for prediction is :', elapsed_time, 'seconds')
    logging.debug("prediction file saved sucessfully")
    print("PREDICTION COMPLETED SUCESSFULLY")

  def idwt(self):
    global wavelet_level
    global wavelet_mode
    global wavelet_type
    global time
    st=time.time()
    cwd_rnn=cwd+"/rnn/"
    logging.debug("IDWT CONVERSION STARTED-RNN")
    print("IDWT CONVERSION STARTED [RNN] ...")
    def get_csv_files(folder):
      csv_files = []
      for file in os.listdir(folder):
          if file.endswith('.csv'):
              csv_files.append(file)
      return csv_files
    my_file1 = open(cwd+"/output_signals.txt", "r")
    # reading the file
    lines1= [line.strip() for line in my_file1] 
    wavelet_type = input("enter the wavelet type :")
    wavelet_mode = input("enter the wavelet mode :")
    wavelet_level= int(input("enter the level of wavlet :"))
    for i in lines1:
      folder1_path = cwd_rnn+"predictions/"+i+"/"
      folder2_path = cwd+"/test/"
      new=pd.DataFrame()
      os.makedirs(cwd_rnn+"predictions_idwt/"+i,exist_ok=True)
      files1 = get_csv_files(folder1_path)
      files2 = get_csv_files(folder2_path)
      for file1 in files1:
        if file1 in files2:
            file1_path = os.path.join(folder1_path, file1)
            file2_path = os.path.join(folder2_path, file1)
            with open(file1_path, 'r') as f1, open(file2_path, 'r') as f2:
                pred= pd.read_csv(f1)
                Time= pd.read_csv(f2)
                actual=Time[i]
                coeffs=pred['predicted_signal']
                sim_time=Time['Sim_time']
                sz =len(Time)
                dummy_signal = [i for i in range(0, sz)]
                coeffs_dummy = pywt.wavedec(dummy_signal,wavelet_type,wavelet_mode,wavelet_level)
                coeff_slices = pywt.coeffs_to_array(coeffs_dummy)[1]
                coeffs_idwt=pywt.array_to_coeffs(coeffs,coeff_slices,output_format='wavedec')
                coeffs_idwt1=pywt.waverec(coeffs_idwt,wavelet_type,wavelet_mode)
                new=pd.DataFrame()
                coeffs_idwt1=coeffs_idwt1[0:sz]
                new['actual_signal']=actual
                new['predicted_signal']=coeffs_idwt1
                new['Sim_time']=sim_time
                new.to_csv(cwd_rnn+'predictions_idwt/'+i+"/"+file1)
      et = time.time()
      elapsed_time = et - st
      print('Execution time for IDWT conversion is :', elapsed_time, 'seconds')
      logging.debug("IDWT conversion sucessfull")
      print("WAVELET TO TIME DOMAIN CONVERSION COMPLETED SUCESSFULLY !!!!")

  def graph_plot(self):
    st = time.time()
    cwd_rnn=cwd+"/rnn/"
    print("Generating Graph Plot")
    my_file1 = open(cwd+"/output_signals.txt", "r")
    lines1 = [line.strip() for line in my_file1]
    for i in lines1:
      path = cwd_rnn+"/predictions_idwt/"+i+"/"+"*.csv"
      lpath = len(path)-5
      list_files = []
      count =0
      os.makedirs(cwd_rnn+"/graphs/"+i,exist_ok=True)
      for file in glob.glob(path):
       count = count+1
       name = file[lpath:]
       name = name[:-4]
       list_files += [name]
       df_fun=pd.read_csv(file)
       X_time=df_fun['Sim_time']
       vinn = df_fun['actual_signal']
       pred_vinn = df_fun['predicted_signal']
       fig=plt.figure(figsize=(16,9),facecolor='w', edgecolor='k')
       plt.plot(X_time,vinn,color="blue", linewidth=3, label = vinn)
       plt.plot(X_time,pred_vinn, color="red", linewidth=3, label = pred_vinn)
       title = name
       plt.xlabel("Time", fontsize=10)
       plt.ylabel("vinn", fontsize=10)
       plt.grid(True)
       plt.legend()
       plt.legend(["Actual signal","Predicted signal"], loc ="lower right")
       plt.title(title)
       nm =cwd_rnn+"/graphs/"+i+"/"+name+".png"
       plt.savefig(nm)
       et = time.time()
    elapsed_time = et - st
    print('Execution time for graph plot is :', elapsed_time, 'seconds')
    logging.debug("graph plotted sucessfull")
    print("GRAPH PLOTTED SUCESSFULLY")

  #calculating metrics using prediction file
  def metrics(self):
    st=time.time()
    cwd_rnn=cwd+"/rnn/"
    logging.debug("metrics calculation started")
    print("METRICS CALCULATION STARTED...")
    my_file1 = open(cwd+"/output_signals.txt", "r")
    lines1 = [line.strip() for line in my_file1]
    for i in lines1:
      file_path = cwd_rnn+"/predictions_idwt/"+i+"/"
      t1=glob.glob(file_path+"*.csv")
      result = pd.DataFrame(columns = ['FILENAME','MAE','MSE','RMSE','R2SCORE','SNR'])
      os.makedirs(cwd_rnn+"/metrics/"+i,exist_ok=True)
      for file in t1:
       filename=file.replace(file_path,"") #read file from path into induvidual files
       filename1= filename.replace(".csv","")
       df=pd.read_csv(file)
       df["diff"] = df["actual_signal"] - df["predicted_signal"]
       length_of_rows=len(df)
       n=length_of_rows-1
       upper=0
       lower=0
       upper = df['actual_signal'].pow(2).sum()
       lower = df['diff'].pow(2).sum()
       upper = upper/n
       lower = lower/n
       snr=10*math.log10(upper/lower)
       predvinn=df['predicted_signal']
       vinn=df['actual_signal']
       rmse=sqrt(mean_squared_error(predvinn,vinn))
       mae = mean_absolute_error(predvinn,vinn)
       mse = mean_squared_error(predvinn,vinn)
       r2=r2_score(predvinn,vinn)
       result= result.append({'FILENAME':filename1,'MAE':mae,'MSE':mse,'RMSE':rmse,'R2SCORE':r2,'SNR':snr},ignore_index=True)
       result_file =cwd_rnn+"/metrics/"+i+"/"+"metrics"+".csv"
       result.to_csv(result_file,index=False)
    et= time.time()
    elapsed_time = et - st
    print('Execution time for metrics calculation is :', elapsed_time, 'seconds')
    logging.debug("metrics calculation completed sucessfully")
    print("METRICS CALCULATION COMPLETED SUCESSFULLY")

class XGBoostModel:
    global cwd_xgb
    global wavelet_level
    global wavelet_mode
    global wavelet_type
    os.makedirs(cwd+"\\XGBoost\\", exist_ok=True)
    cwd_xgb = cwd+"\\XGBoost\\"

    def model_creation(self):
        logging.debug("Model Creation started")
        print("XGBoost Model Creation Started.....")
        st = time.time()
        stime = datetime.datetime.now()
        print("XGBoost Model creation start time:-", stime)
        combined_csv = pd.read_csv(cwd + "\\combined_train_wavelet.csv")
        os.makedirs(cwd_xgb+"\\models", exist_ok=True)

        with open(cwd+"/input_signals.txt", "r") as f:
            inputs = [line.strip() for line in f]

        with open(cwd+"/output_signals.txt", "r") as f:
            outputs = [line.strip() for line in f]

        X = combined_csv[inputs].values
        est = int(input("Enter the number of estimators for XGBoost: "))

        for i in outputs:
            print("Creating model for signal -", i)
            y = combined_csv[i].values
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

            model = XGBRegressor(n_estimators=est, n_jobs=-1, verbosity=0)
            print("Model fitting")
            model.fit(X_train, y_train)
            print("Fitting completed")

            y_train_predicted = model.predict(X_train)
            y_test_predicted = model.predict(X_test)
            mse_train = mean_squared_error(y_train, y_train_predicted)
            mse_test = mean_squared_error(y_test, y_test_predicted)

            print("XGBoost, Train MSE: {} Test MSE: {}".format(mse_train, mse_test))
            model_path = cwd_xgb + "/models/signal_" + i + ".joblib"
            joblib.dump(model, open(model_path, 'wb'), compress=3)

        et = time.time()
        etime = datetime.datetime.now()
        print("Model creation end time:-", etime)
        print('Execution time for creating model is:', et - st, 'seconds')
        logging.debug("Model saved successfully")
        print("XGBOOST MODEL SAVED SUCCESSFULLY")

    def prediction(self):
        st = time.time()
        print("XGBOOST PREDICTION STARTED...")
        logging.debug("Prediction started")
        stime = datetime.datetime.now()
        print("Prediction start time:-", stime)

        file_path = cwd + "\\test_wav\\"
        t1 = glob.glob(file_path + "*.csv")

        with open(cwd+"/input_signals.txt", "r") as f:
            inputs = [line.strip() for line in f]
        with open(cwd+"/output_signals.txt", "r") as f:
            outputs = [line.strip() for line in f]

        for file in t1:
            filename = file.replace(file_path, "")
            filename1 = filename.replace(".csv", "")
            df = pd.read_csv(file)

            for i in outputs:
                row = df[inputs].values
                model_path = cwd_xgb + "\\models/signal_" + i + ".joblib"
                model = joblib.load(open(model_path, 'rb'))

                logging.debug("Trained model loaded successfully")
                os.makedirs(cwd_xgb + "\\predictions\\" + i, exist_ok=True)

                yhat = model.predict(row)
                name = pd.DataFrame()
                name['Sim_time'] = df['Sim_time']
                name['predicted_signal'] = yhat
                name['actual_signal'] = df[i]
                name.to_csv(cwd_xgb + "\\predictions\\" + i + "\\" + filename1 + ".csv")

        et = time.time()
        print("Prediction end time:-", datetime.datetime.now())
        print('Execution time for prediction is:', et - st, 'seconds')
        logging.debug("Prediction file saved successfully")
        print("PREDICTION COMPLETED SUCCESSFULLY")

    def idwt(self):
        global wavelet_level, wavelet_mode, wavelet_type
        st = time.time()
        logging.debug("IDWT conversion started - XGB")
        print("IDWT CONVERSION STARTED [XGB] ...")

        def get_csv_files(folder):
            return [file for file in os.listdir(folder) if file.endswith('.csv')]

        with open(cwd+"/output_signals.txt", "r") as f:
            lines1 = [line.strip() for line in f]

        wavelet_type = input("Enter the wavelet type: ")
        wavelet_mode = input("Enter the wavelet mode: ")
        wavelet_level = int(input("Enter the wavelet level: "))

        for i in lines1:
            folder1_path = cwd_xgb + "predictions/" + i + "/"
            folder2_path = cwd + "/test/"
            os.makedirs(cwd_xgb + "predictions_idwt/" + i, exist_ok=True)

            files1 = get_csv_files(folder1_path)
            files2 = get_csv_files(folder2_path)

            for file1 in files1:
                if file1 in files2:
                    file1_path = os.path.join(folder1_path, file1)
                    file2_path = os.path.join(folder2_path, file1)

                    pred = pd.read_csv(file1_path)
                    Time = pd.read_csv(file2_path)

                    actual = Time[i]
                    coeffs = pred['predicted_signal']
                    sim_time = Time['Sim_time']
                    sz = len(Time)
                    dummy_signal = [j for j in range(0, sz)]
                    coeffs_dummy = pywt.wavedec(dummy_signal, wavelet_type, wavelet_mode, wavelet_level)
                    coeff_slices = pywt.coeffs_to_array(coeffs_dummy)[1]
                    coeffs_idwt = pywt.array_to_coeffs(coeffs, coeff_slices, output_format='wavedec')
                    coeffs_idwt1 = pywt.waverec(coeffs_idwt, wavelet_type, wavelet_mode)
                    coeffs_idwt1 = coeffs_idwt1[:sz]

                    new = pd.DataFrame()
                    new['actual_signal'] = actual
                    new['predicted_signal'] = coeffs_idwt1
                    new['Sim_time'] = sim_time
                    new.to_csv(cwd_xgb + 'predictions_idwt\\' + i + "\\" + file1)

        print('Execution time for IDWT conversion is:', time.time() - st, 'seconds')
        logging.debug("IDWT conversion successful")
        print("WAVELET TO TIME DOMAIN CONVERSION COMPLETED SUCCESSFULLY !!!!")

    def graph_plot(self):
        st = time.time()
        print("Generating Graph Plot with Metrics")

        with open(cwd + "/output_signals.txt", "r") as f:
            lines1 = [line.strip() for line in f]

        for i in lines1:
            path = cwd_xgb + "/predictions_idwt/" + i + "/" + "*.csv"
            lpath = len(path) - 5
            os.makedirs(cwd_xgb + "/graphs/" + i, exist_ok=True)

            for file in glob.glob(path):
                name = file[lpath:-4]
                df_fun = pd.read_csv(file)
                X_time = df_fun['Sim_time']
                actual = df_fun['actual_signal']
                predicted = df_fun['predicted_signal']

                df_fun["diff"] = actual - predicted
                n = len(df_fun) - 1
                upper = actual.pow(2).sum() / n
                lower = df_fun['diff'].pow(2).sum() / n
                snr = 10 * math.log10(upper / lower) if lower > 0 else float('inf')

                mae = mean_absolute_error(actual, predicted)
                mse = mean_squared_error(actual, predicted)
                rmse = sqrt(mse)
                r2 = r2_score(predicted, actual)

                fig, ax = plt.subplots(figsize=(16, 9))
                ax.plot(X_time, actual, color="blue", linewidth=2, label="Actual Signal")
                ax.plot(X_time, predicted, color="red", linewidth=2, label="Predicted Signal")

                metrics_text = f"MAE: {mae:.4f}\nMSE: {mse:.4f}\nRMSE: {rmse:.4f}\nR²: {r2:.4f}\nSNR: {snr:.4f} dB"
                ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, fontsize=12,
                        verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='lightgray'))

                ax.set_title(name, fontsize=14)
                ax.set_xlabel("Time", fontsize=12)
                ax.set_ylabel("Pll_out", fontsize=12)
                ax.grid(True)
                ax.legend(loc="lower right")

                plt.savefig(cwd_xgb + "/graphs/" + i + "/" + name + ".png")
                plt.close()

        print('Execution time for graph plot is:', time.time() - st, 'seconds')
        logging.debug("Graph plotted successfully")
        print("GRAPH PLOTTED SUCCESSFULLY")

    def metrics(self):
        st = time.time()
        logging.debug("Metrics calculation started")
        print("METRICS CALCULATION STARTED...")

        with open(cwd+"/output_signals.txt", "r") as f:
            lines1 = [line.strip() for line in f]

        for i in lines1:
            file_path = cwd_xgb + "/predictions_idwt/" + i + "/"
            t1 = glob.glob(file_path + "*.csv")
            result = pd.DataFrame(columns=['FILENAME', 'MAE', 'MSE', 'RMSE', 'R2SCORE', 'SNR'])
            os.makedirs(cwd_xgb + "/metrics/" + i, exist_ok=True)

            for file in t1:
                filename1 = file.replace(file_path, "").replace(".csv", "")
                df = pd.read_csv(file)
                df["diff"] = df["actual_signal"] - df["predicted_signal"]
                n = len(df) - 1
                upper = df['actual_signal'].pow(2).sum() / n
                lower = df['diff'].pow(2).sum() / n
                snr = 10 * math.log10(upper / lower) if lower > 0 else float('inf')

                pred = df['predicted_signal']
                actual = df['actual_signal']
                rmse = sqrt(mean_squared_error(actual, pred))
                mae = mean_absolute_error(actual, pred)
                mse = mean_squared_error(actual, pred)
                r2 = r2_score(actual, pred)

                result = pd.concat([result, pd.DataFrame([{
                    'FILENAME': filename1, 'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2SCORE': r2, 'SNR': snr
                }])], ignore_index=True)

            result.to_csv(cwd_xgb + "/metrics/" + i + "/metrics.csv", index=False)

        print('Execution time for metrics calculation is:', time.time() - st, 'seconds')
        logging.debug("Metrics calculation completed successfully")
        print("METRICS CALCULATION COMPLETED SUCCESSFULLY")

class lstm_new:
  def model_creation(self):
    st = time.time()
    global cwd_lstm
    os.makedirs(cwd+"/lstm/",exist_ok=True)
    cwd_lstm=cwd+"/lstm/"
    print("LSTM Model creation Started")
    os.makedirs(cwd_lstm+"/models",exist_ok=True)
    logging.debug("files in train folder combined for modeling")
    combined_csv=pd.read_csv(cwd+"/combined_train_wavelet.csv",chunksize=50000)
    combined_csv=pd.concat(combined_csv)
    my_file = open(cwd+"/input_signals.txt", "r")
    #reading file
    lines = [line.strip() for line in my_file]
    logging.debug("signals from input file are read sucessfully")
    #reading output signals
    my_file1 = open(cwd+"/output_signals.txt", "r")
    # reading the file
    lines1 = [line.strip() for line in my_file1]
    logging.debug("signals from output file are read sucessfully")
    inputs=lines
    X=combined_csv[inputs].values
    X= np.asarray(X)
    outputs=lines1
    for i in outputs:
      os.makedirs(cwd_lstm+"/models/"+i+"/",exist_ok=True)
      y=combined_csv[i].values
      y= np.asarray(y)
      length=len(combined_csv)
      in_len=len(inputs)
      X = X.reshape(length,1,in_len)
      y = array(y)
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
      model = Sequential()
      model.add(LSTM(10,activation='relu',return_sequences=True, input_shape = (1,in_len)))
      model.add(LSTM(32,activation='relu'))
      model.add(Dense(1))
      model.compile(optimizer=Adam(lr=0.001), loss='mse')
      history=model.fit(x=X_train,y=y_train,validation_data=(X_test,y_test),batch_size=32,epochs=20)
      model.save(cwd_lstm+"/models/"+i+"/")
      plt.plot(history.history['loss'])
      plt.plot(history.history['val_loss'])
      plt.title('model loss')
      plt.ylabel('loss')
      plt.xlabel('epoch')
      plt.legend(['train', 'val'], loc='upper left')
      plt.savefig(cwd_lstm+"/models/"+i+"/"+"model_graph_"+i+".png")
      et = time.time()
      elapsed_time = et - st
      print('Execution time for creating model is :', elapsed_time, 'seconds')
      logging.debug("model saved sucessfully")
      print("LSTM MODEL SAVED SUCESSFULLY")

  def prediction(self):
    st = time.time()
    cwd_lstm=cwd+"/lstm/"
    print("LSTM PREDICTION STARTED...")
    logging.debug("prediction started")
    file_path=cwd+"/test_wav/"
    t1=glob.glob(file_path+"*.csv")
    output=pd.DataFrame()
    for file in t1:
      filename=file.replace(file_path,"") #read file from path into induvidual files
      filename1= filename.replace(".csv","")
      df=pd.read_csv(file)
      logging.debug("signals from input file are read sucessfully for prediction")
      my_file = open(cwd+"/input_signals.txt", "r")
      # reading the file
      lines = [line.strip() for line in my_file]
      my_file1 = open(cwd+"/output_signals.txt", "r")
      # reading the file
      lines1 = [line.strip() for line in my_file1]
      logging.debug("signals from output file are read sucessfully for prediction")
      outputs=lines1
      inputs=lines
      length=len(df)
      in_len=len(inputs)
      for i in outputs:
       my_model = keras.models.load_model(cwd_lstm+"/models/"+i+"/")
       os.makedirs(cwd_lstm+"/predictions/"+i+"/",exist_ok=True)
       row=df[inputs].values
       row = row.reshape(length,1,in_len)
       logging.debug("trained model loaded sucessfully")
       prediction=my_model.predict(row,verbose=0)
       print(prediction)
       prediction = [item for sublist in prediction for item in sublist]
       name="output"+i
       name=pd.DataFrame()
       name['Sim_time']=df['Sim_time']
       name['predicted_signal']=prediction#acess numpy array colum wise
       name['actual_signal']=df[i]
       print(name)
       name.to_csv(cwd_lstm+"/predictions/"+i+"/"+filename1+".csv")
    et = time.time()
    elapsed_time = et - st
    print('Execution time for prediction is :', elapsed_time, 'seconds')
    logging.debug("prediction file saved sucessfully")
    print("PREDICTION COMPLETED SUCESSFULLY")
  
  def idwt(self):
    global time
    st=time.time()
    logging.debug("IDWT CONVERSION STARTED")
    cwd_lstm=cwd+"/lstm/"
    print("IDWT CONVERSION STARTED [LSTM]...")
    def get_csv_files(folder):
      csv_files = []
      for file in os.listdir(folder):
          if file.endswith('.csv'):
              csv_files.append(file)
      return csv_files
    my_file1 = open(cwd+"/output_signals.txt", "r")
    # reading the file
    lines1= [line.strip() for line in my_file1] 
    for i in lines1:
      folder1_path = cwd_lstm+"predictions/"+i+"/"
      print(folder1_path)
      folder2_path = cwd+"/test/"
      print(folder2_path)
      new=pd.DataFrame()
      os.makedirs(cwd_lstm+"predictions_idwt/"+i,exist_ok=True)
      files1 = get_csv_files(folder1_path)
      files2 = get_csv_files(folder2_path)
      wavelet_type = input("enter the wavelet type :")
      wavelet_mode = input("enter the wavelet mode :")
      wavelet_level= int(input("enter the level of wavlet :"))
      for file1 in files1:
        if file1 in files2:
            file1_path = os.path.join(folder1_path, file1)
            file2_path = os.path.join(folder2_path, file1)
            with open(file1_path, 'r') as f1, open(file2_path, 'r') as f2:
                pred= pd.read_csv(f1)
                Time= pd.read_csv(f2)
                actual=Time[i]
                coeffs=pred['predicted_signal']
                sim_time=Time['Sim_time']
                sz =len(Time)
                dummy_signal = [i for i in range(0, sz)]
                coeffs_dummy = pywt.wavedec(dummy_signal,wavelet_type,wavelet_mode,wavelet_level)
                coeff_slices = pywt.coeffs_to_array(coeffs_dummy)[1]
                coeffs_idwt=pywt.array_to_coeffs(coeffs,coeff_slices,output_format='wavedec')
                coeffs_idwt1=pywt.waverec(coeffs_idwt,wavelet_type,wavelet_mode)
                new=pd.DataFrame()
                coeffs_idwt1=coeffs_idwt1[0:sz]
                new['actual_signal']=actual
                new['predicted_signal']=coeffs_idwt1
                new['Sim_time']=sim_time
                new.to_csv(cwd_lstm+'predictions_idwt/'+i+'/'+file1)
      et = time.time()
      elapsed_time = et - st
      print('Execution time for IDWT conversion is :', elapsed_time, 'seconds')
      logging.debug("IDWT conversion sucessfull")
      print("WAVELET TO TIME DOMAIN CONVERSION COMPLETED SUCESSFULLY !!!!")

  def graph_plot(self):
    st = time.time()
    cwd_lstm=cwd+"/lstm/"
    print("Generating Graph Plot")
    my_file1 = open(cwd+"/output_signals.txt", "r")
    lines1 = [line.strip() for line in my_file1]
    for i in lines1:
      path = cwd_lstm+"/predictions_idwt/"+i+"/"+"*.csv"
      lpath = len(path)-5
      list_files = []
      count =0
      os.makedirs(cwd_lstm+"/graphs/"+i,exist_ok=True)
      for file in glob.glob(path):
       count = count+1
       name = file[lpath:]
       name = name[:-4]
       list_files += [name]
       df_fun=pd.read_csv(file)
       X_time=df_fun['Sim_time']
       vinn = df_fun['actual_signal']
       pred_vinn = df_fun['predicted_signal']
       fig=plt.figure(figsize=(16,9),facecolor='w', edgecolor='k')
       plt.plot(X_time,vinn,color="blue", linewidth=3, label = vinn)
       plt.plot(X_time,pred_vinn, color="red", linewidth=3, label = pred_vinn)
       title = name
       plt.xlabel("Time", fontsize=10)
       plt.ylabel("vinn", fontsize=10)
       plt.grid(True)
       plt.legend()
       plt.legend(["Actual signal","Predicted signal"], loc ="lower right")
       plt.title(title)
       nm =cwd_lstm+"/graphs/"+i+"/"+name+".png"
       plt.savefig(nm)
      et = time.time()
    elapsed_time = et - st
    print('Execution time for graph plot is :', elapsed_time, 'seconds')
    logging.debug("graph plotted sucessfull")
    print("GRAPH PLOTTED SUCESSFULLY")

  #calculating metrics using prediction file
  def metrics(self):
    st=time.time()
    cwd_lstm=cwd+"/lstm/"
    logging.debug("metrics calculation started")
    print("METRICS CALCULATION STARTED...")
    my_file1 = open(cwd+"/output_signals.txt", "r")
    lines1 = [line.strip() for line in my_file1]
    for i in lines1:
      file_path = cwd_lstm+"/predictions_idwt/"+i+"/"
      t1=glob.glob(file_path+"*.csv")
      result = pd.DataFrame(columns = ['FILENAME','MAE','MSE','RMSE','R2SCORE','SNR'])
      os.makedirs(cwd_lstm+"/metrics/"+i,exist_ok=True)
      for file in t1:
       filename=file.replace(file_path,"") #read file from path into induvidual files
       filename1= filename.replace(".csv","")
       df=pd.read_csv(file)
       df["diff"] = df["actual_signal"] - df["predicted_signal"]
       length_of_rows=len(df)
       n=length_of_rows-1
       upper=0
       lower=0
       upper = df['actual_signal'].pow(2).sum()
       lower = df['diff'].pow(2).sum()
       upper = upper/n
       lower = lower/n
       snr=10*math.log10(upper/lower)
       predvinn=df['predicted_signal']
       vinn=df['actual_signal']
       rmse=sqrt(mean_squared_error(predvinn,vinn))
       mae = mean_absolute_error(predvinn,vinn)
       mse = mean_squared_error(predvinn,vinn)
       r2=r2_score(predvinn,vinn)
       result= result.append({'FILENAME':filename1,'MAE':mae,'MSE':mse,'RMSE':rmse,'R2SCORE':r2,'SNR':snr},ignore_index=True)
       result_file =cwd_lstm+"/metrics/"+i+"/"+"metrics"+".csv"
       result.to_csv(result_file,index=False)
    et= time.time()
    elapsed_time = et - st
    print('Execution time for metrics calculation is :', elapsed_time, 'seconds')
    logging.debug("metrics calculation completed sucessfully")
    print("METRICS CALCULATION COMPLETED SUCESSFULLY")
    
if __name__ == "__main__":

  print('''   
                  __  __    _         _         _    _ _ _ _   _ _ _ _ _ 
                 |  \/  |  | |       | |   _   | |  |  _ _ _| |_ _   _ _|
                 | \  / |  | |       | |  / \  | |  | |           | |
                 | |\/| |  | |       | | /   \ | |  | |           | |
                 | |  | |  | |____   | |/ / \ \| |  | |_ _ _      | |
                 |_|  |_|  |______|  |__ /   \ __|  |_ _ _ _|     |_|     \n ''')
  print('''                 *** MACHINE LEARNING BASED AMS WAVEFORM PREDICTION SCRIPT *** \n \n              
	         MACHINE LEARNING AMS CIRCUIT MODELLING USING WAVELET TRANSFORM\n
                 1. RANDOM FOREST MODELLING  \n
                 2. MULTI-LAYER PERCEPTRON MODELLING \n
                 3. LSTM MODELLING \n
                 4. RNN MODELLING \n
                 5. ALL ALGORITHMS MODELLING \n
                 6. XGBOOST
                 7. EXIT PROGRAM  \n ''')
  choice = int(input("Select an option: "))
  rf= RandomForest()
  multi=MLP()
  light=RNN()
  lstm=lstm_new()
  pr=preprocess()
  xg = XGBoostModel()
  os.makedirs(cwd+"/Memory_usage/",exist_ok=True)
  cwd_mem=cwd+"/Memory_usage/"
  if choice == 1:
    st_all=time.time()
    print("RANDOM FOREST MODELLING SELECTED")
    with open(cwd_mem+"usage.txt", "w") as f:
      f.write("###### RESOURCE USAGE STATS #######")
      f.write("\n")
      vcc=psutil.cpu_count()
      f.write('Total number of CPUs in server:')
      f.write(str(vcc))
      f.write("\n")
      f.write("Total memory of server in Gbs:")
      mem = psutil.virtual_memory()[0]
      mem=mem/1024/1024/1024
      f.write(str(mem))
      f.write("\n")
    pr.specs("DWT conversion process")
    pr.dwt_train()
    pr.dwt_test()
    pr.specs("BEFORE ALL PROCESS")
    pr.combine_files()
    pr.specs("AFTER COMBINING FILES")
    pr.fetch()
    pr.specs("AFTER FETCHING DATA")
    rf.model_creation()
    pr.specs("AFTER MODEL CREATION")
    rf.prediction()
    pr.specs("AFTER PREDICTION")
    rf.idwt()
    rf.graph_plot()
    rf.metrics()
    et_all= time.time()
    elapsed_time = et_all - st_all
    print('Complete Execution time for random forest Modelling is :', elapsed_time, 'seconds')
  elif choice == 2:
    st_all=time.time()
    with open(cwd_mem+"usage.txt", "w") as f:
      f.write("##MLP MODELLING STATS##")
      f.write("###### RESOURCE USAGE STATS #######")
      f.write("\n")
      vcc=psutil.cpu_count()
      f.write('Total number of CPUs in server:')
      f.write(str(vcc))
      f.write("\n")
      f.write("Total memory of server in Gbs:")
      mem = psutil.virtual_memory()[0]
      mem=mem/1024/1024/1024
      f.write(str(mem))
      f.write("\n")
    print("MLP MODELLING SELECTED")
    pr.specs("BEFORE ALL PROCESS-MLP")
    pr.dwt_train()
    pr.dwt_test()
    pr.combine_files()
    pr.specs("AFTER COMBINING FILES-MLP")
    multi.model_creation()
    pr.specs("AFTER MODEL CREATION")
    multi.prediction()
    pr.specs("AFTER PREDICTION")
    multi.idwt()
    multi.graph_plot()
    multi.metrics()
    et_all= time.time()
    elapsed_time = et_all - st_all
    print('Complete Execution time for MLP is :', elapsed_time, 'seconds')
  elif choice == 3:
    st_all=time.time()
    with open(cwd_mem+"usage.txt", "w") as f:
      f.write("##LSTM MODELLING STATS##")
      f.write("###### RESOURCE USAGE STATS #######")
      f.write("\n")
      vcc=psutil.cpu_count()
      f.write('Total number of CPUs in server:')
      f.write(str(vcc))
      f.write("\n")
      f.write("Total memory of server in Gbs:")
      mem = psutil.virtual_memory()[0]
      mem=mem/1024/1024/1024
      f.write(str(mem))
      f.write("\n")
    print("LSTM MODELLING SELECTED")
    pr.specs("BEFORE ALL PROCESS-LSTM")
    #pr.dwt_train()
    #pr.dwt_test()
    #pr.combine_files()
    pr.specs("AFTER COMBINING FILES-LSTM")
    pr.fetch()
    lstm.model_creation()
    pr.specs("AFTER MODEL CREATION-LSTM")
    lstm.prediction()
    pr.specs("AFTER PREDICTION-LSTM")
    lstm.idwt()
    lstm.graph_plot()
    lstm.metrics()
    et_all= time.time()
    elapsed_time = et_all - st_all
    print('Complete Execution time for LSTM is :', elapsed_time, 'seconds')
  elif choice == 4:
    st_all=time.time()
    with open(cwd_mem+"usage.txt", "w") as f:
      f.write("##RNN MODELLING STATS##")
      f.write("###### RESOURCE USAGE STATS #######")
      f.write("\n")
      vcc=psutil.cpu_count()
      f.write('Total number of CPUs in server:')
      f.write(str(vcc))
      f.write("\n")
      f.write("Total memory of server in Gbs:")
      mem = psutil.virtual_memory()[0]
      mem=mem/1024/1024/1024
      f.write(str(mem))
      f.write("\n")
    print("RNN MODELLING SELECTED")
    pr.specs("BEFORE ALL PROCESS-RNN")
    pr.dwt_train()
    pr.dwt_test()
    pr.combine_files()
    pr.specs("AFTER COMBINING FILES-RNN")
    #pr.fetch()
    #light.model_creation()
    pr.specs("AFTER MODEL CREATION-RNN")
    #light.prediction()
    pr.specs("AFTER PREDICTION-RNN")
    #light.idwt()
    #light.graph_plot()
    #light.metrics()
    et_all= time.time()
    elapsed_time = et_all - st_all
    print('Complete Execution time for RNN is :', elapsed_time, 'seconds')  
  elif choice== 5:
    st_all=time.time()
    with open(cwd_mem+"usage.txt", "w") as f:
      f.write("##ALL MODELLING STATS##")
      f.write("###### RESOURCE USAGE STATS #######")
      f.write("\n")
      vcc=psutil.cpu_count()
      f.write('Total number of CPUs in server:')
      f.write(str(vcc))
      f.write("\n")
      f.write("Total memory of server in Gbs:")
      mem = psutil.virtual_memory()[0]
      mem=mem/1024/1024/1024
      f.write(str(mem))
      f.write("\n")
    print("ALL ML MODELLING SELECTED")
    pr.dwt_train()
    pr.dwt_test()
    pr.specs("BEFORE ALL PROCESS-ALL")
    pr.combine_files()
    pr.specs("AFTER COMBINING FILES-ALL")
    pr.fetch()
    pr.specs("AFTER FETCHING COMBINED FILES-ALL")
    rf.model_creation()
    pr.specs("AFTER MODEL CREATION-RF")
    multi.model_creation()
    pr.specs("AFTER MODEL CREATION-MLP")
    lstm.model_creation()
    light.model_creation()
    pr.specs("AFTER MODEL CREATION-LSTM")
    rf.prediction()
    pr.specs("AFTER PREDICTION-RF")
    multi.prediction()
    pr.specs("AFTER PREDICTION-MLP")
    lstm.prediction()
    light.prediction()
    pr.specs("AFTER PREDICTION-LSTM")
    rf.idwt()
    multi.idwt()
    lstm.idwt()
    light.idwt()
    rf.graph_plot()
    multi.graph_plot()
    lstm.graph_plot()
    light.graph_plot()
    rf.metrics()
    multi.metrics()
    lstm.metrics()
    light.metrics()
    et_all= time.time()
    elapsed_time = et_all - st_all
    print('Complete Execution time for all algorithm is :', elapsed_time, 'seconds')
  elif choice == 6:
    st_all=time.time()
    with open(cwd_mem+"usage.txt", "w") as f:
      f.write("##XGBOST MODELLING STATS##")
      f.write("###### RESOURCE USAGE STATS #######")
      f.write("\n")
      vcc=psutil.cpu_count()
      f.write('Total number of CPUs in server:')
      f.write(str(vcc))
      f.write("\n")
      f.write("Total memory of server in Gbs:")
      mem = psutil.virtual_memory()[0]
      mem=mem/1024/1024/1024
      f.write(str(mem))
      f.write("\n")
    print("XGBOOST MODELLING SELECTED")
    pr.specs("BEFORE ALL PROCESS-XGBOOST")
    pr.dwt_train()
    pr.dwt_test()
    pr.combine_files()
    pr.specs("AFTER COMBINING FILES-XGBOOST")
    xg.model_creation()
    pr.specs("AFTER MODEL CREATION")
    xg.prediction()
    pr.specs("AFTER PREDICTION")
    xg.idwt()
    xg.graph_plot()
    xg.metrics()
    et_all= time.time()
    elapsed_time = et_all - st_all
    print('Complete Execution time for MLP is :', elapsed_time, 'seconds')
elif choice == 7:
    sys.exit()




            
