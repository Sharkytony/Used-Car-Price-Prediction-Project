from flask import Flask, render_template,request
import numpy as np
import pipeline
import pickle

with open('color_encoder.pkl', 'rb')as file:
    color_encoder = pickle.load(file)

with open('dw_encoder.pkl', 'rb')as file:
    dw_encoder = pickle.load(file)

with open('gbt_encoder.pkl', 'rb')as file:
    gbt_encoder = pickle.load(file) 

with open('fuel_encoder.pkl', 'rb')as file:
    fuel_encoder = pickle.load(file)

with open('cat_encoder.pkl', 'rb')as file:
    cat_encoder = pickle.load(file)
    
with open('man_encoder.pkl', 'rb')as file:
    man_encoder = pickle.load(file)

with open('scaler.pkl', 'rb')as file :
    scaler = pickle.load(file)

with open('rforest_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/preds', methods=['GET', 'POST'])
def prediction():
    new_input =request.form.to_dict()
    if new_input['Turbo'] == 'Yes':
        new_input['Engine volume'] = new_input['Engine volume'] + ' Turbo'
    else :
        pass

    input_df = pipeline.create_df(np.NaN, new_input['Levy'], new_input['Manufacturer'], np.NaN, new_input['Prod. year'], new_input['Category'],new_input['Leather interior'], new_input['Fuel type'], new_input['Engine volume'] , new_input['Mileage'],new_input['Cylinders'],new_input['Gear box type'], new_input['Drive wheels'], np.NaN, new_input['Wheel'], new_input['Color'], new_input['Airbags'])

    input_processed_df = pipeline.entire_pipeline(input_df, man_encoder, cat_encoder,fuel_encoder, gbt_encoder, dw_encoder,color_encoder, scaler)

    pred_price = loaded_model.predict(input_processed_df)
    return render_template('predictions.html', predicted_price=round(pred_price[0]))

if __name__ == '__main__':
    app.run(debug=True)