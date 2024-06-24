from flask import (
    Flask,
    render_template,
    url_for,
    redirect
)
from forms import InputForms
import joblib
from ml_model import column_selector,le
import pandas as pd 

app = Flask(__name__)
app.config["SECRET_KEY"] = "this_is_seret_key"

model = joblib.load("model.joblib")

@app.route('/')
@app.route('/home')
def home():
    form = InputForms()
    return render_template("home.html",title='home',form=form) 

@app.route('/predict',methods=["GET","POST"])
def predict():
    form = InputForms()
    if form.validate_on_submit():
        x_new = pd.DataFrame({
            "Application mode" : [form.Application_mode.data],
            "Tuition fees up to date":[form.tution_fee_up_to_date.data],
            "Gender":[form.Gender.data],
            "Scholarship holder":[form.Scholarship_holder.data],
            "Age at enrollment":[form.Age_at_enrollment.data],
            "Curricular units 1st sem (evaluations)":[form.Curricular_units_1st_sem_evaluation.data],
            "Curricular units 1st sem (approved)":[form.Curricular_units_1st_sem_approved.data],
            "Curricular units 1st sem (grade)":[form.Curricular_units_1st_sem_grade.data],
            "Curricular units 2nd sem (enrolled)":[form.Curricular_units_2nd_sem_enrolled.data],
            "Curricular units 2nd sem (evaluations)":[form.Curricular_units_2st_sem_evaluation.data],
            "Curricular units 2nd sem (approved)":[form.Curricular_units_2nd_sem_approved.data],
            "Curricular units 2nd sem (grade)":[form.Curricular_units_2nd_sem_grade.data]
        })
        prediction = model.predict(x_new)
        message = f"The prediction  is {le.inverse_transform(prediction)}"
    else:
        message = "Please provide valid input details!"
    return render_template('predict.html',title="Predict",form=form,output=message)  


if __name__ == "__main__":
    app.run(debug=True) 