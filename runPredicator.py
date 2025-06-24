#importing the required libraries

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS  # ✅ import this

app = Flask(__name__)
CORS(app)  # ✅ enable CORS for all routes


@app.route("/predict-run",methods=['POST'])
def run_predictor():
    data=request.get_json()
    balls=data['balls']
    year=data['year']
    home=data['home']
    against=data['against']
    pitch_type=data['pitch_type']
    innings=data['innings']
    virat_data= pd.read_csv('virat.csv')
    #binart encoding
    home_code_map={'Home':0,'Away':1}
    virat_data['HomeCode']=virat_data['Home_Away'].map(home_code_map)
    #multi hot encoding
    enc=OneHotEncoder(sparse_output=False,handle_unknown='ignore')
    enc.fit(virat_data[['Pitch Type','Against']])
    encoded_array=enc.transform(virat_data  [['Pitch Type','Against']])
    encoded_cols=enc.get_feature_names_out(['Pitch Type','Against'])
    #making the data frames of the hot encoded data
    encoded_df=pd.DataFrame(encoded_array,columns=encoded_cols,index=virat_data.index)
    #final dataframes 
    virat_data=pd.concat([virat_data,encoded_df],axis=1)    
    #features for the inputs
    features=['Balls','HomeCode','Year','Innings'] + list(encoded_cols)
    X=virat_data[features] #inputs
    Y=virat_data['Runs'] #ouputs
    #Now make a linear regression models
    model = LinearRegression()
    model.fit(X,Y) #the model fits a line on the given input and output
    #now make the dataset of the given inputs
    user_df=pd.DataFrame([[pitch_type,against]],columns=['Pitch Type','Against'])
    user_encoded = enc.transform(user_df) #transfored into the hot-encoded format
    user_encoded_df=pd.DataFrame(user_encoded,columns=encoded_cols)
    #appling the binary encoding in the home/away user input 
    home_code=home_code_map.get(home,0)
    #make the combined data frames of the input data
    input_data= pd.DataFrame([[balls,home_code,year,innings]],columns=['Balls','HomeCode','Year','Innings'])
    final_input_df=pd.concat([input_data,user_encoded_df],axis=1)
    final_input_df=final_input_df[features]
    predicated_run=model.predict(final_input_df)[0]
    if predicated_run<0:
        predicated_run=0
    
    return jsonify({'result':int(predicated_run)})
if __name__=='__main__':
    app.run(debug=True,port=4000)




    



