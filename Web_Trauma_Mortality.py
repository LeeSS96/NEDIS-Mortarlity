from asyncore import compact_traceback
from flask_bootstrap import Bootstrap
from wtforms import Form, TextAreaField, FloatField, IntegerField, SelectField, StringField, validators
# from wtforms.validators import DataRequired, Required, ValidationError, input_required

import numpy as np
import pandas as pd
from tensorflow import keras
import pickle as cPickle
from Web_Trauma_Mortality.Input_Function import define_model, eval_model_DNN





# intention_accident, intention_suicide, intention_assult, intention_others, intention_unknown, intention_missing_data, arcs_traffic_car, arcs_traffic_bike, arcs_traffic_motorcycle, arcs_traffic_others, arcs_traffic_unknown, arcs_fall, onehot_Damagemechanism[6], onehot_Damagemechanism[7], onehot_Damagemechanism[8], onehot_Damagemechanism[9], onehot_Damagemechanism[10], onehot_Damagemechanism[11], onehot_Damagemechanism[12], onehot_Damagemechanism[13], onehot_Damagemechanism[14], onehot_Damagemechanism[15], onehot_Damagemechanism[16] = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
# alert, drowsy, semicoma, coma, unknown_response, initial_severity_1, onehot_Initialseveritytriageresult[1], onehot_Initialseveritytriageresult[2], onehot_Initialseveritytriageresult[3], onehot_Initialseveritytriageresult[4], initial_severity_etc, initial_severity_unknown, initial_severity_missing_data = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
# altar_severity_1, onehot_Changedseveritytriageresult[1], onehot_Changedseveritytriageresult[2], onehot_Changedseveritytriageresult[3], onehot_Changedseveritytriageresult[4], altar_severity_etc, altar_severity_missing_data, emergency, non_emergency, emergency_unknown = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

### https://medium.com/better-programming/how-to-use-flask-wtforms-faab71d5a034
### https://getbootstrap.com/docs/4.0/components/card/
### https://www.w3schools.com/bootstrap/bootstrap_tables.asp

class Trauma_MortalityForm(Form):
    
    Age = SelectField("Age",
                      choices=[('1', '1: Under 1'), ('2', '2: 1~4'), ('3', '3: 5~9'), ('4', '4: 10~14'),('5', '5: 15~19'), ('6', '6: 20~24'), ('7', '7: 25~29'), ('8', '8: 30~34'),
                               ('9', '9: 35~39'), ('10', '10: 40~44'), ('11', '11: 45~49'), ('12', '12: 50~54'),('13', '13: 55~59'), ('14', '14: 60~64'), ('15', '15: 65~69'), ('16', '16: 70~74'),
                               ('17', '17: 75~79'), ('18', '18: 80~84'), ('19', '19: 85~89'), ('20', '20: 90~94'),('21', '21: 95~99'), ('22', '22: 100~104'), ('23', '23: 105~109'), ('24', '24: 110~114'),
                               ('25', '25: 115~119'), ('26', '26: over 120')])
    Gender = SelectField("Gender", choices=[('0', 'Female'), ('1', 'Male')])
    Intentionality = SelectField('Intentionality', choices=[('0', '1: accidental, unintentional'), ('1', '2: self-harm, suicide'), ('2', '3: violence, assault'), ('3', '4: other specified'), ('4', '5: unspecified'), ('5', '6: data missing')])
    Damagemechanism = SelectField('Injury mechanism', choices=[('0', '1: traffic accident-car'), ('1', '2: traffic accident-bike'), ('2', '3: traffic accident-motorcycle'), ('3', '4: traffic accident-etc.'), ('4', '5: traffic accident-unspecified'), ('5', '6: fall'), ('6', '7: slip down'), ('7', '8: struck'), ('8', '9: firearm/cut/pierce'), ('9', '10: machine'), ('10', '11: fire, flames or heat'), ('11', '12: drowning or nearly'), ('12', '13: poisoning'), ('13', '14: choking, hanging'), ('14', '15: etc.'), ('15', '16: unknown'), ('16', '17: data missing')])
    Emergencysymptoms = SelectField('Emergent symptoms', choices=[('0', 'Yes'), ('1', 'No'), ('2', 'Unknown')])
    Reactionafterhospitalized = SelectField('AVPU scale', choices=[('0', 'Alert'), ('1', 'Verbal response(drowsy)'), ('2', 'Painful response(semicoma)'), ('3', 'Unresponsive(coma)'), ('4', 'Unknown')])
    Initialseveritytriageresult = SelectField('Initial KTAS', choices=[('0', '1: Level 1(Resuscitation)'), ('1', '2: Level 2(Emergency)'), ('2', '3: Level 3(Urgency)'), ('3', '4: Level 4(Less urgency)'), ('4', '5: Level 5(Nonurgency)'), ('5', '6: unknown'), ('6', '7: data missing')])
    Changedseveritytriageresult = SelectField('Altered KTAS', choices=[('0', '1: Level 1(Resuscitation)'), ('1', '2: Level 2(Emergency)'), ('2', '3: Level 3(Urgency)'), ('3', '4: Level 4(Less urgency)'), ('4', '5: Level 5(Nonurgency)'), ('5', '6: data missing')])
    Torsosurgerychest = SelectField('Torso procedure-chest', choices=[('0', 'No'), ('1', 'Yes')])
    Torsosurgeryabdomen = SelectField('Torso procedure-abdomen', choices=[('0', 'No'), ('1', 'Yes')])
    Torsosurgeryvascular = SelectField('Torso procedure-vascular', choices=[('0', 'No'), ('1', 'Yes')])
    Torsosurgeryheart = SelectField('Torso procedure-heart', choices=[('0', 'No'), ('1', 'Yes')])
    Headsurgery = SelectField('Head surgery', choices=[('0', 'No'), ('1', 'Yes')])
    Ecmo = SelectField('ECMO', choices=[('0', 'No'), ('1', 'Yes')])
    ICD = StringField("Three-digit ICD-10 code (ex. S072, S224, T083)")

def function_one_hot_encoding(arrValue, arrSize):
    arrData = np.zeros((arrSize))
    if arrValue != 'None':
        arrData[int(arrValue)] = 1
    return arrData

def function_one_hot_encoding2(arrValue, arrSize):
    arrData = np.zeros((arrSize))
    if arrValue != 'None':
        arrData[int(arrValue)] = 1
    arrData = np.delete(arrData, arrSize - 1)
    return arrData

def method_evaluation(input_eval_data):
    SavePath = './Web_Trauma_Mortality/DNN/'

    model_dnn = define_model()
    prob_dnn = eval_model_DNN(input_eval_data, model_dnn, SavePath)
    keras.backend.clear_session()  # Clear the model in memory

    pred = np.where(prob_dnn > 0.401464, 1, 0).squeeze()
    deceased_rate = round(prob_dnn * 100, 4)
    survive_rate = 100 - deceased_rate

    return survive_rate, deceased_rate, pred

def send_list_values(Age, Gender, Intentionality, Damagemechanism, Emergencysymptoms, Reactionafterhospitalized, Initialseveritytriageresult, Changedseveritytriageresult, ICD):
    SavePath = './Web_Trauma_Mortality/DNN/'
    ICD10 = pd.read_table(SavePath + '3digit_ICD_10_code.txt',sep=',',low_memory=False)
    list_Age = ['1: Under 1', '2: 1~4', '3: 5~9', '4: 10~14', '5: 15~19', '6: 20~24', '7: 25~29', '8: 30~34', \
                '9: 35~39', '10: 40~44', '11: 45~49', '12: 50~54', '13: 55~59', '14: 60~64', '15: 65~69', '16: 70~74', \
                '17: 75~79', '18: 80~84', '19: 85~89', '20: 90~94', '21: 95~99', '22: 100~104', '23: 105~109', '24: 110~114', \
                '25: 115~119', '26: over 120']
    send_Age = list_Age[int(Age)-1]

    list_Gender = ['Female', 'Male']
    send_Gender = list_Gender[int(Gender)]

    list_Intentionality = ['1: accidental, unintentional', '2: self-harm, suicide', '3: violence, assault', '4: other specified', '5: unspecified', '6: data missing']
    send_Intentionality = list_Intentionality[int(Intentionality)]

    list_Damagemechanism = ['1: traffic accident-car', '2: traffic accident-bike', '3: traffic accident-motorcycle', '4: traffic accident-etc.', '5: traffic accident-unspecified', '6: fall', '7: slip down', '8: struck', '9: firearm/cut/pierce', '10: machine', '11: fire, flames or heat', '12: drowning or nearly', '13: poisoning', '14: choking, hanging', '15: etc.', '16: unknown', '17: data missing']
    send_Damagemechanism = list_Damagemechanism[int(Damagemechanism)]

    list_Emergencysymptoms = ['Yes', 'No', 'Unknown']
    send_Emergencysymptoms = list_Emergencysymptoms[int(Emergencysymptoms)]

    list_Reactionafterhospitalized = ['Alert', 'Verbal response(drowsy)', 'Painful response(semicoma)', 'Unresponsive(coma)', 'Unknown']
    send_Reactionafterhospitalized = list_Reactionafterhospitalized[int( Reactionafterhospitalized)]

    list_Initialseveritytriageresult = ['1: Level 1(Resuscitation)', '2: Level 2(Emergency)', '3: Level 3(Urgency)', '4: Level 4(Less urgency)', '5: Level 5(Nonurgency)', '6: unknown', '7: data missing']
    send_Initialseveritytriageresult = list_Initialseveritytriageresult[int(Initialseveritytriageresult)]

    list_Changedseveritytriageresult = ['1: Level 1(Resuscitation)', '2: Level 2(Emergency)', '3: Level 3(Urgency)', '4: Level 4(Less urgency)', '5: Level 5(Nonurgency)', '6: data missing']
    send_Changedseveritytriageresult = list_Changedseveritytriageresult[int(Changedseveritytriageresult)]

    ICD = ICD.replace(", ", ",")
    send_ICD10 = ICD.split(",")
    for code in send_ICD10:
        if code in ICD10:
            ICD10[code] = 1
    list_ICD = ICD10.loc[0].tolist()

    return send_Age, send_Gender, send_Intentionality, send_Damagemechanism, send_Emergencysymptoms, send_Reactionafterhospitalized, send_Initialseveritytriageresult, send_Changedseveritytriageresult, send_ICD10, list_ICD

    


def send_bool_values(Torsosurgerychest, Torsosurgeryabdomen, Torsosurgeryvascular, Torsosurgeryheart, Headsurgery, Ecmo):
    input_data = np.concatenate([[Torsosurgerychest], [Torsosurgeryabdomen], [Torsosurgeryvascular], [Torsosurgeryheart], [Headsurgery], [Ecmo]])
    send_data = np.asarray(['No' if input_data[ii] == '0' else 'Yes' for ii in range(0, 6)])
    return send_data

def processTrauma_MortalityForm(request):
    form = Trauma_MortalityForm(request.form)
    return form

def processTrauma_MortalityResult(request):

    # global intention_accident, intention_suicide, intention_assult, intention_others, intention_unknown, intention_missing_data, arcs_traffic_car, arcs_traffic_bike, arcs_traffic_motorcycle, arcs_traffic_others, arcs_traffic_unknown, arcs_fall, onehot_Damagemechanism[6], onehot_Damagemechanism[7], onehot_Damagemechanism[8], onehot_Damagemechanism[9], onehot_Damagemechanism[10], onehot_Damagemechanism[11], onehot_Damagemechanism[12], onehot_Damagemechanism[13], onehot_Damagemechanism[14], onehot_Damagemechanism[15], onehot_Damagemechanism[16]
    # global alert, drowsy, semicoma, coma, unknown_response, initial_severity_1, onehot_Initialseveritytriageresult[1], onehot_Initialseveritytriageresult[2], onehot_Initialseveritytriageresult[3], onehot_Initialseveritytriageresult[4], initial_severity_etc, initial_severity_unknown, initial_severity_missing_data
    # global altar_severity_1, onehot_Changedseveritytriageresult[1], onehot_Changedseveritytriageresult[2], onehot_Changedseveritytriageresult[3], onehot_Changedseveritytriageresult[4], altar_severity_etc, altar_severity_missing_data, emergency, non_emergency, emergency_unknown
    
    # intention_accident, intention_suicide, intention_assult, intention_others, intention_unknown, intention_missing_data, arcs_traffic_car, arcs_traffic_bike, arcs_traffic_motorcycle, arcs_traffic_others, arcs_traffic_unknown, arcs_fall, onehot_Damagemechanism[6], onehot_Damagemechanism[7], onehot_Damagemechanism[8], onehot_Damagemechanism[9], onehot_Damagemechanism[10], onehot_Damagemechanism[11], onehot_Damagemechanism[12], onehot_Damagemechanism[13], onehot_Damagemechanism[14], onehot_Damagemechanism[15], onehot_Damagemechanism[16] = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    # alert, drowsy, semicoma, coma, unknown_response, initial_severity_1, onehot_Initialseveritytriageresult[1], onehot_Initialseveritytriageresult[2], onehot_Initialseveritytriageresult[3], onehot_Initialseveritytriageresult[4], initial_severity_etc, initial_severity_unknown, initial_severity_missing_data = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    # altar_severity_1, onehot_Changedseveritytriageresult[1], onehot_Changedseveritytriageresult[2], onehot_Changedseveritytriageresult[3], onehot_Changedseveritytriageresult[4], altar_severity_etc, altar_severity_missing_data, emergency, non_emergency, emergency_unknown = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

    form = Trauma_MortalityForm(request.form)
    Age = request.form.get('Age')
    Gender = request.form['Gender']
    Intentionality = request.form['Intentionality']
    onehot_Intentionality = function_one_hot_encoding2(Intentionality, arrSize=6)
    Damagemechanism = request.form['Damagemechanism']
    onehot_Damagemechanism = function_one_hot_encoding2(Damagemechanism, arrSize=17)
    Emergencysymptoms = request.form['Emergencysymptoms']
    onehot_Emergencysymptoms = function_one_hot_encoding(Emergencysymptoms, arrSize=3)
    Reactionafterhospitalized = request.form['Reactionafterhospitalized']
    onehot_Reactionafterhospitalized = function_one_hot_encoding(Reactionafterhospitalized, arrSize=5)
    Initialseveritytriageresult = request.form['Initialseveritytriageresult']
    onehot_Initialseveritytriageresult = function_one_hot_encoding2(Initialseveritytriageresult, arrSize=7)
    Changedseveritytriageresult = request.form['Changedseveritytriageresult']
    onehot_Changedseveritytriageresult = function_one_hot_encoding2(Changedseveritytriageresult, arrSize=6)
    Torsosurgerychest = request.form['Torsosurgerychest']
    Torsosurgeryabdomen = request.form['Torsosurgeryabdomen']
    Torsosurgeryvascular = request.form['Torsosurgeryvascular']
    Torsosurgeryheart = request.form['Torsosurgeryheart']
    Headsurgery = request.form['Headsurgery']
    Ecmo = request.form['Ecmo']
    ICD = request.form['ICD']


    

    ###==================================================================================###
    send_Age, send_Gender, send_Intentionality, send_Damagemechanism, send_Emergencysymptoms, send_Reactionafterhospitalized, send_Initialseveritytriageresult, send_Changedseveritytriageresult, send_ICD, list_ICD = send_list_values(Age, Gender, Intentionality, Damagemechanism, Emergencysymptoms, Reactionafterhospitalized, Initialseveritytriageresult, Changedseveritytriageresult, ICD)
    send_bool = send_bool_values(Torsosurgerychest, Torsosurgeryabdomen, Torsosurgeryvascular, Torsosurgeryheart, Headsurgery, Ecmo)

    Age = (int(Age)-1)/25
    
    input_data = np.concatenate(
        [[Age], [Headsurgery], [Torsosurgeryvascular], [Torsosurgeryabdomen], [Torsosurgerychest], [Torsosurgeryheart], [Ecmo], onehot_Initialseveritytriageresult, onehot_Changedseveritytriageresult, onehot_Intentionality,
         onehot_Damagemechanism, onehot_Emergencysymptoms, onehot_Reactionafterhospitalized, [Gender], list_ICD])
    input_data = np.asarray([np.NaN if input_data[ii] == '' else input_data[ii] for ii in range(0, len(input_data))], dtype=np.float64)

    print('Input Data: ', input_data)
    print()

    input_data = input_data.reshape(1, -1)
    input_data = np.asarray(input_data, dtype=np.float64)
    rate_survive, rate_Mortality, person_status = method_evaluation(input_data)
    ###==================================================================================###

    return form, send_Age, send_bool[4], send_bool[2], send_bool[1], send_bool[0], send_bool[3], send_bool[5], send_Initialseveritytriageresult, send_Changedseveritytriageresult, send_Intentionality, \
         send_Damagemechanism, send_Emergencysymptoms, send_Reactionafterhospitalized, send_Gender, send_ICD, rate_survive, rate_Mortality, person_status