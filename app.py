from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

### HOME
@app.route('/')
def home() :
    return render_template('index.html')

### HALAMAN DATASET
@app.route('/dataset', methods = ['GET', 'POST'])
def dataset() :
    return render_template('dataset.html')

### HALAMAN LAPORAN HASIL ANALISA
@app.route('/report', methods = ['GET', 'POST'])
def report() :
    return render_template('report.html')

### HALAMAN INPUT DATA UNTUK PREDIKSI LIFE EXPECTANCY
@app.route('/regression', methods = ['GET', 'POST'])
def regression() :
    return render_template('regression.html')

## HALAMAN HASIL PREDIKSI LIFE EXPECTANCY
@app.route('/reg_result', methods = ['GET', 'POST'])
def reg_result() :
    if request.method == 'POST' :
        input = request.form

        df_predict = pd.DataFrame({
            'country' : [ input['country'] ],
            'year' : [ input['year'] ],
            'adult_mortality' : [ input['adult_mortality'] ],
            '_bmi_' : [ input['_bmi_'] ],
            'income_composition_of_resources' : [ input['income_composition_of_resources'] ],
            'schooling' : [ input['schooling'] ],
            'percentage_expenditure' : [ input['percentage_expenditure'] ],
            'under-five_deaths_' : [ input['under-five_deaths_'] ],
            '_hiv/aids' : [ input['_hiv/aids'] ]
        })

        prediksi_regresi = round(reg_model.predict(df_predict)[0], 2)

        print(prediksi_regresi)

    return render_template('reg_result.html', data=input, pred = prediksi_regresi)

### HALAMAN INPUT DATA UNTUK PREDIKSI COUNTRY STATUS
@app.route('/classification', methods = ['GET', 'POST'])
def classification() :
    return render_template('classification.html')

## HALAMAN HASIL PREDIKSI COUNTRY STATUS
@app.route('/class_result', methods = ['GET', 'POST'])
def class_result() :
    if request.method == 'POST' :
        input = request.form

        df_predict = pd.DataFrame({
            'country' : [ input['country'] ],
            'year' : [ input['year'] ],
            'alcohol' : [ input['alcohol'] ],
            'schooling' : [ input['schooling'] ],
            'life_expectancy_' : [ input['life_expectancy_'] ],
            'percentage_expenditure' : [ input['percentage_expenditure'] ]
        })

        prediksi_clasifikasi = round(log_model.predict_proba(df_predict)[0][1] * 100, 1)

        print(prediksi_clasifikasi)

    return render_template('class_result.html', data=input, pred = prediksi_clasifikasi)


### HALAMAN ABOUT
@app.route('/about', methods = ['GET', 'POST'])
def about() :
    return render_template('about.html')





if __name__ == '__main__' :

    reg_model = pd.read_pickle(r'regression_model.sav')
    log_model = pd.read_pickle(r'classification_model.sav')
    app.run(debug=True)