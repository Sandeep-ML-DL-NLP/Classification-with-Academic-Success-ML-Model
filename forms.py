from flask_wtf import FlaskForm 
from wtforms import (
    StringField,
    SubmitField,
    IntegerField,
    SelectField,
    FloatField
)

from wtforms.validators import DataRequired,NumberRange 
import pandas as pd 
cols = ['Application mode', 'Tuition fees up to date', 'Gender',
       'Scholarship holder', 'Age at enrollment',
       'Curricular units 1st sem (evaluations)',
       'Curricular units 1st sem (approved)',
       'Curricular units 1st sem (grade)',
       'Curricular units 2nd sem (enrolled)',
       'Curricular units 2nd sem (evaluations)',
       'Curricular units 2nd sem (approved)',
       'Curricular units 2nd sem (grade)']

train_data = pd.read_csv(r'C:\Users\Cars24\Downloads\Classification with an Academic Success\data\train.csv')

class InputForms(FlaskForm):
    Application_mode = SelectField(
        label="Application mode",
        choices=train_data["Application mode"].unique().tolist(),
        validators=[DataRequired()]
    )
    tution_fee_up_to_date = SelectField(
        label="Tuition fees up to date",
        choices=train_data["Tuition fees up to date"].unique().tolist(),
        validators=[DataRequired()]
    )
    Gender = SelectField(
        label="Gender",
        choices=train_data["Gender"].unique().tolist(),
        validators=[DataRequired()]
    )
    Scholarship_holder = SelectField(
        label="Scholarship holder",
        choices=train_data["Scholarship holder"].unique().tolist(),
        validators=[DataRequired()]
    )
    Age_at_enrollment = IntegerField(
        label="Age at enrollment",
        validators=[DataRequired(),
        NumberRange(min=10, max=80, message="Age must be between 10 and 80")
        ]
    )
    Curricular_units_1st_sem_evaluation = IntegerField(
        label="Curricular units 1st sem (evaluations)",
        validators=[DataRequired(),
        NumberRange(min=1, max=50, message="Number must be between 1 and 50")
        ])
    Curricular_units_1st_sem_approved = IntegerField(
        label="Curricular units 1st sem (approved)",
        validators=[DataRequired(),
        NumberRange(min=1, max=30, message="Number must be between 1 and 30")
        ]
    )
    Curricular_units_1st_sem_grade = FloatField(
        label="Curricular units 1st sem (grade)",
        validators=[DataRequired(),
        NumberRange(min=1, max=30, message="float values must be between 1 and 30")
        ]
    )
    Curricular_units_2st_sem_evaluation = IntegerField(
        label="Curricular units 2nd sem (evaluations)",
        validators=[DataRequired(),
        NumberRange(min=1, max=50, message="Number must be between 1 and 50")
        ])
    Curricular_units_2nd_sem_approved = IntegerField(
        label="Curricular units 2nd sem (approved)",
        validators=[DataRequired(),
        NumberRange(min=1, max=30, message="Number must be between 1 and 30")
        ]
    )
    Curricular_units_2nd_sem_grade = FloatField(
        label="Curricular units 2nd sem (grade)",
        validators=[DataRequired(),
        NumberRange(min=1, max=30, message="float values must be between 1 and 30")
        ]
    )
    Curricular_units_2nd_sem_enrolled = FloatField(
        label="Curricular units 2nd sem (enrolled)",
        validators=[DataRequired(),
        NumberRange(min=1, max=30, message="float values must be between 1 and 30")
        ]
    )
    submit = SubmitField("Predict")