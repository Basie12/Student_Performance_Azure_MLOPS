from flask import Flask, request, render_template
import sys
from src.pipelines.predict_pipeline import CustomData, PredictPipeline
from src.exception import CustomException
from src.logger import logging

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('home.html', results=None)

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html', results=None)
    
    try:
        data = CustomData(
            gender=request.form.get('gender'),
            ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('reading_score')),
            writing_score=float(request.form.get('writing_score'))
        )
        pred_df = data.get_data_as_data_frame()
        logging.info(f"Input DataFrame: {pred_df.to_dict()}")

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('home.html', results=round(float(results[0]), 2))
    
    except CustomException as e:
        logging.error(f"Prediction error: {str(e)}")
        return render_template('home.html', results=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(host='0.0.0.0')