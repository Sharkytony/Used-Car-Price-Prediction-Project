import pandas as pd

def create_df(id, levy, man, mod, yr, cat, leather, fuel, eng, mil, cy, gear, dw, doors, steer, col, airbags):

    data = pd.DataFrame({
        'ID': [id],
        'Levy': [levy],
        'Manufacturer': [man],
        'Model': [mod],
        'Prod. year': [yr],
        'Category': [cat],
        'Leather interior': [leather],
        'Fuel type': [fuel],
        'Engine volume': [eng],
        'Mileage': [mil],
        'Cylinders': [cy],
        'Gear box type': [gear],
        'Drive wheels': [dw],
        'Doors': [doors],
        'Wheel': [steer],
        'Color': [col],
        'Airbags': [airbags]
    })
    
    return data

def check_df(df):
    sum = 0
    cols = ['ID', 'Levy', 'Manufacturer', 'Model', 'Prod. year','Category', 'Leather interior', 'Fuel type', 'Engine volume', 'Mileage','Cylinders',
    'Gear box type', 'Drive wheels', 'Doors', 'Wheel', 'Color','Airbags']

    allowed_null = ['ID', 'Model', 'Doors']

    if df.shape[1] == 17:
        for col1,col2 in zip(df.columns, cols):
            if col1 == col2:
                sum +=1
                if col1 not in allowed_null and df[col1].isnull().sum() > 0 :
                    return f'Missing important data in "{col1}" column '
                else :
                    df[col1] = df[col1].fillna(0)

    else :
        return 'Incorrect data format'
    
    if len(cols) == sum:
        df.reset_index(inplace=True)
        df.drop(columns=['index'], inplace=True)
        return df
    else : 
        return 'Incorrect data format'
    
def drop_features(df):
    features = ['ID', 'Model', 'Doors']
    df.drop(columns=features,inplace=True)
    return df

def clean_Levy(df):
    def levy_handler(val):
        try:
            val = int(val)
        except Exception:
            val = 0
        return val
    df['Levy'] = df['Levy'].apply(levy_handler)
    df['Levy'] = df['Levy'].astype(int)
    return df

def clean_Mileage(df):
    def mileage_handler(val):
        if 'km' in val : 
            val = int(val[:-2].strip())
        else :
            pass
        return val
    
    df['Mileage']= df['Mileage'].apply(mileage_handler)
    return df

def clean_EngVol(df):
    def eng_vol_handler():
        turbos = list()
        values = list()
        for val in df['Engine volume']:
            try : 
                val = float(val)
                turbo = 'No'
            except:
                val = val.split(' ')[0]
                turbo = 'Yes'
            turbos.append(turbo)
            values.append(val)
        return turbos,values
    df['Turbo'], df['Engine Volume']= eng_vol_handler()
    df.drop(columns=['Engine volume'], inplace=True)
    df['Engine Volume'] = df['Engine Volume'].astype(float)

    return df

def clean_Wheel(df):
    def wheel_handler(val):
        if 'right' in val.lower():
            return 1
        else :
            return 0

    df['Wheel'] = df['Wheel'].apply(wheel_handler)
    return df

def clean_Color(df):
    color_uni = ['Silver', 'Black', 'White', 'Grey', 'Blue', 'Green', 'Red', 'Sky blue', 'Orange', 'Yellow', 'Brown', 'Golden', 'Beige', 'Carnelian red', 'Purple', 'Pink']

    color_vc ={'Black': 0, 'White': 0, 'Silver': 0, 'Grey': 0, 'Blue': 0, 'Red': 0, 'Green': 0, 'Brown': 0, 'Carnelian red': 136, 'Golden': 120, 'Sky blue': 106, 'Beige': 102, 'Yellow': 94, 'Orange': 71, 'Purple': 23, 'Pink': 21}
    
    new_colors = dict()

    for item in df['Color']:
        if item not in color_uni:
            new_colors[item] = 'Others'

    for key, val in color_vc.items():
        if val > 0:
            new_colors[key] = 'Others'
        else : 
            new_colors[key] = key

    df['Color'] = df['Color'].map(new_colors)
    return df

def clean_FuelType(df):
    fuel_uni = ['Petrol', 'Diesel', 'Hydrogen', 'Hybrid', 'Plug-in Hybrid', 'LPG', 'CNG']
    fuel_map = dict()
    for i in df['Fuel type']:
        if i not in fuel_uni:
            fuel_map[i] = 'Other'
        elif i == 'Plug-in Hybrid' or i == 'Hydrogen':
            fuel_map[i] = 'Other'
        else:
            fuel_map[i] = i
            
    df['Fuel type'] = df['Fuel type'].map(fuel_map)
    return df

def clean_Category(df):
    cat_uni = ['Jeep', 'Hatchback', 'Sedan', 'Microbus', 'Goods wagon', 'Universal', 'Coupe', 'Minivan', 'Cabriolet', 'Limousine', 'Pickup']

    cat_map = {'Sedan': 'Sedan', 'Jeep': 'Jeep', 'Hatchback': 'Hatchback',
 'Minivan': 'Minivan', 'Coupe': 'Coupe', 'Universal': 'Others', 'Microbus': 'Others', 'Goods wagon': 'Others', 'Pickup': 'Others', 'Cabriolet': 'Others',
 'Limousine': 'Others'}
    
    for category in df['Category']:
        if category not in cat_uni:
            cat_map[category] = 'Others'

    
    df['Category'] = df['Category'].map(cat_map)
    return df

def clean_Manufacturer(df):
    def lowercase(row_val):
        return row_val.lower().strip()
    df['Manufacturer'] = df['Manufacturer'].apply(lowercase)

    man_uni = ['lexus', 'chevrolet', 'honda', 'ford', 'hyundai', 'toyota', 'mercedes-benz', 'opel', 'porsche', 'bmw', 'jeep', 'volkswagen', 'audi', 'renault','nissan', 'subaru', 'daewoo', 'kia', 'mitsubishi', 'ssangyong', 'mazda', 'gmc','fiat', 'infiniti', 'alfa romeo', 'suzuki', 'acura', 'lincoln', 'vaz', 'gaz','citroen', 'land rover', 'mini', 'dodge', 'chrysler', 'jaguar', 'isuzu', 'skoda','daihatsu', 'buick', 'tesla', 'cadillac', 'peugeot', 'bentley', 'volvo', 'სხვა','haval', 'hummer', 'scion', 'uaz', 'mercury', 'zaz', 'rover', 'seat', 'lancia','moskvich', 'maserati', 'ferrari', 'saab', 'lamborghini', 'rolls-royce', 'pontiac', 'saturn', 'aston martin', 'greatwall']

    others = ['bmw', 'nissan', 'lexus', 'volkswagen', 'ssangyong', 'kia', 'subaru', 'audi', 'mitsubishi', 'opel', 'mazda', 'daewoo', 'jeep', 'fiat', 'suzuki', 'mini', 'dodge', 'land rover', 'renault', 'jaguar', 'skoda', 'chrysler', 'porsche', 'peugeot', 'buick', 'vaz', 'infiniti', 'volvo', 'acura', 'citroen', 'gmc', 'scion', 'lincoln', 'cadillac', 'alfa romeo', 'mercury', 'daihatsu', 'maserati', 'gaz', 'saab', 'seat', 'other', 'lancia', 'haval', 'hummer', 'pontiac', 'saturn', 'greatwall']

    man_map = dict()
    
    for val in df['Manufacturer']:
        if val not in man_uni :
            man_map[val] = 'other'
        elif val in others:
            man_map[val] = 'other'
        else : 
            man_map[val] = val

    df['Manufacturer'] = df['Manufacturer'].map(man_map)
    return df

def clean_ProdYr(df):
    def calc_age(year):
        year = int(year)
        age = 2024 - year
        return age
    df['Car_Age'] = df['Prod. year'].apply(calc_age)
    df.drop(columns=['Prod. year'], inplace=True)
    return df

def clean_Turbo(df):
    df['Turbo'] = df['Turbo'].map({'Yes':1.0,'No':0.0})
    return df

def clean_Lint(df):
    df['Leather interior'] = df['Leather interior'].map({'Yes':1.0,
                                                                 'No':0.0})
    return df

def encoding(df, man_encoder, cat_encoder,fuel_encoder, 
             gbt_encoder, dw_encoder,color_encoder):
    man_pred = man_encoder.transform(df[['Manufacturer']])
    feature_names = man_encoder.get_feature_names_out(['Manufacturer'])
    man_pred = pd.DataFrame(man_pred, columns=feature_names)

    cat_pred = cat_encoder.transform(df[['Category']])
    feature_names = cat_encoder.get_feature_names_out(['Category'])
    cat_pred = pd.DataFrame(cat_pred, columns=feature_names)

    fuel_pred = fuel_encoder.transform(df[['Fuel type']])
    feature_names = fuel_encoder.get_feature_names_out(['Fuel type'])
    fuel_pred = pd.DataFrame(fuel_pred, columns=feature_names)

    gbt_pred = gbt_encoder.transform(df[['Gear box type']])
    feature_names = gbt_encoder.get_feature_names_out(['Gear box type'])
    gbt_pred = pd.DataFrame(gbt_pred, columns=feature_names)

    dw_pred = dw_encoder.transform(df[['Drive wheels']])
    feature_names = dw_encoder.get_feature_names_out(['Drive wheels'])
    dw_pred = pd.DataFrame(dw_pred, columns=feature_names)

    col_pred = color_encoder.transform(df[['Color']])
    feature_names = color_encoder.get_feature_names_out(['Color'])
    col_pred = pd.DataFrame(col_pred, columns=feature_names)
    
    df.drop(columns=['Manufacturer', 'Category', 'Fuel type', 'Gear box type',
                     'Drive wheels', 'Color'], inplace=True)
    df = pd.concat([df, man_pred, cat_pred, fuel_pred, gbt_pred,dw_pred, col_pred], axis=1)
    return df

def scaling(df,scaler):
    scaling_cols =['Levy','Mileage','Cylinders','Airbags','Engine Volume','Car_Age']
    scaled_df = pd.DataFrame(scaler.transform(df[scaling_cols]), columns=scaling_cols)
    df.drop(columns=scaling_cols, inplace=True)
    df = pd.concat([df, scaled_df], axis=1)
    return df

def entire_pipeline(df, man_encoder, cat_encoder,fuel_encoder, 
    gbt_encoder, dw_encoder,color_encoder,scaler):
    
    data = check_df(df)
    data = drop_features(data)
    data = clean_Levy(data)
    data = clean_Mileage(data)
    data = clean_EngVol(data)
    data = clean_Wheel(data)
    data = clean_Color(data)
    data = clean_FuelType(data)
    data = clean_Category(data)
    data = clean_Manufacturer(data)
    data = clean_ProdYr(data)
    data = clean_Turbo(data)
    data = clean_Lint(data)
    data = encoding(data, man_encoder, cat_encoder,fuel_encoder, 
             gbt_encoder, dw_encoder,color_encoder)
    data = scaling(data,scaler)

    return data