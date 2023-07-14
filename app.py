from flask import Flask, make_response, jsonify, render_template, session, send_file, request, send_from_directory, app, send_from_directory, Response
from flask_restx import Resource, Api, reqparse, fields
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
from flask_jwt_extended import create_access_token
import jwt
import shutil
import traceback
import os
import io
import time
from keras.applications.mobilenet import preprocess_input
from keras.applications.mobilenet import decode_predictions
import random
from flask_mail import Mail, Message
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename
import numpy as np
import cv2
import tensorflow as tf
from keras.models import load_model
import mysql.connector
from PIL import Image
from keras.preprocessing import image
import matplotlib.pyplot as plt
from werkzeug.datastructures import FileStorage
import werkzeug
import tempfile

app = Flask(__name__)  # Instantiation of Flask object.
api = Api(app)        # Instantiation of Flask-RESTX object.
CORS(app)
# Path untuk menyimpan video
UPLOAD_FOLDER = 'C:\\Users\\ASUS\\Downloads\\flask-deteksi-kemacetan\\upload'

############################
##### BEGIN: Database #####
##########################
app.config["SQLALCHEMY_DATABASE_URI"] = "mysql://root:@127.0.0.1:3306/deteksi_kemacetan_kendaraan"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'WhatEverYouWant'
app.config['MAIL_SERVER'] = 'smtp.gmail.com'  # mail env config
app.config['MAIL_PORT'] = 465
app.config['MAIL_USE_SSL'] = True
app.config['MAIL_USERNAME'] = "redsihelma@gmail.com"
app.config['MAIL_PASSWORD'] = "eqgvdirjjzumolyp"
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Misalnya, 16MB
# api.add_resource(PredictVideo, '/predict-video')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}

mail = Mail(app)
db = SQLAlchemy(app)  # Instantiation of Flask-SQLAlchemy object.


# Model User
class User(db.Model):
    id = db.Column(db.Integer(), primary_key=True, nullable=False)
    firstname = db.Column(db.String(35), nullable=False)
    lastname = db.Column(db.String(35), nullable=False)
    email = db.Column(db.String(65), unique=True, nullable=False)
    password = db.Column(db.String(123), nullable=False)
    is_verified = db.Column(db.Boolean(1), nullable=False)
    createdAt = db.Column(db.Date)
    updatedAt = db.Column(db.Date)
    

# Model Video
class Video(db.Model):
    id = db.Column(db.Integer(), primary_key=True, nullable=False)
    filename = db.Column(db.String(10), nullable=False)
    path = db.Column(db.String(50), nullable=False)

class Prediction(db.Model):
    __tablename__ = "predictions"  # Table name used in the database
    id = db.Column(db.Integer, primary_key=True)
    prediction = db.Column(db.String(100))

    def __init__(self, prediction):
        self.prediction = prediction


# Definisikan ekstensi file yang diperbolehkan
ALLOWED_EXTENSIONS = {'mp4', 'avi'}

# Load model deteksi kemacetan
model = load_model('model.h5')

# Define the class labels
classes_list = ["heavy", "light", "medium"]

parser = reqparse.RequestParser()
parser.add_argument(
    'video', type=werkzeug.datastructures.FileStorage, location='files')

##########################
##### END: Database #####
########################

###########################
##### BEGIN: Upload video #####
#########################

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@api.route('/upload-video', methods=['POST'])
class UploadAndPredictVideo(Resource):
    @api.expect(api.parser().add_argument('video', location='files', type=FileStorage, required=True))
    def post(self):
        if 'video' not in request.files:
            return {'message': 'Video tidak ditemukan'}, 400

        video_file = request.files['video']

        # Pastikan nama file dan format file sesuai
        if video_file.filename == '':
            return {'message': 'Nama file tidak valid'}, 400
        if not allowed_file(video_file.filename):
            return {'message': 'Format file video tidak didukung'}, 400

        # Simpan video ke direktori yang ditentukan
        filename = secure_filename(video_file.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        video_file.save(video_path)

        # Predict video
        output_video_path = os.path.join(tempfile.gettempdir(), 'output.mp4')
        result = predict_video(video_path, output_video_path)

        if result['success']:
            try:
                # Mengirimkan video output
                return send_file(output_video_path, as_attachment=True)

            except Exception as e:
                return {'message': 'Gagal mengirimkan video output', 'error': str(e)}, 500
        else:
            return {'message': 'Gagal menghasilkan video output', 'error': result['message']}, 500


def predict_video(input_path, output_path):
    try:
        # Load your pre-trained MobileNetV1 model
        model = load_model('model.h5')

        # Load video
        video = cv2.VideoCapture(input_path)

        # Get video properties
        frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = video.get(cv2.CAP_PROP_FPS)

        # Define output video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        output_video = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        # Process each frame in the video
        while video.isOpened():
            ret, frame = video.read()

            if not ret:
                break

            # Convert frame to gray
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Preprocess the frame for input to the model
            processed_frame = cv2.resize(gray_frame, (224, 224))
            processed_frame = processed_frame.astype('float32') / 255.0
            processed_frame = np.repeat(processed_frame[..., np.newaxis], 3, axis=-1)  # Ubah dimensi keempat menjadi 3 saluran

            # Make predictions on the frame
            predictions = model.predict(np.expand_dims(processed_frame, axis=0))
            top_prediction = np.argmax(predictions[0])

            # Map class index to label
            class_labels = ["heavy", "light", "medium"]
            class_label = class_labels[top_prediction]

            # Display the predicted class on the frame
            cv2.putText(frame, class_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Write the frame to the output video
            output_video.write(frame)

        # Release video capture and video writer
        video.release()
        output_video.release()

        return {'success': True}

    except Exception as e:
        return {'success': False, 'message': str(e)}


###########################
##### END: Upload video #####
#########################

###########################
##### BEGIN: predict video realtime #####
#########################
    
camera = None
is_playing = False
video_width = 1280
video_height = 720
video_fps = 5
video_writer = None
output_filename = 'output.mp4'
prediction_list = []  # Daftar prediksi
previous_prediction_time = time.time()  # Waktu prediksi sebelumnya


# Load the pretrained Mobilenetv1 model
model = tf.keras.models.load_model('model.h5')
classes_list = ["heavy", "light", "medium"]


def preprocess_image(frame):
    # Preprocess the frame for Mobilenetv1 model
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    frame = cv2.resize(frame, (224, 224))
    frame = tf.keras.applications.mobilenet.preprocess_input(frame)
    frame = np.expand_dims(frame, axis=0)
    return frame

def predict_frame(frame):
    # Preprocess the frame
    preprocessed_frame = preprocess_image(frame)

    # Perform prediction on the frame using the Mobilenetv1 model
    predictions = model.predict(preprocessed_frame)
    predicted_class = classes_list[np.argmax(predictions[0])]
    return predicted_class

def add_prediction_text_to_frame(frame, prediction):
    # Add the prediction text to the frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 3
    font_thickness = 4
    text_size = cv2.getTextSize(prediction, font, font_scale, font_thickness)[0]
    text_x = 10  # Position the text at the top-left corner of the frame
    text_y = 30
    cv2.putText(frame, prediction, (text_x, text_y), font, font_scale, (0, 255, 0), font_thickness)

def open_camera():
    global camera, video_writer
    try:
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, video_width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, video_height)
        time.sleep(2)  # Wait for a few seconds to ensure the camera is fully opened
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Use MJPG codec for video output
        video_writer = cv2.VideoWriter(output_filename, fourcc, video_fps, (video_width, video_height))
        if not camera.isOpened():
            raise Exception("Failed to open camera.")
    except Exception as e:
        print(str(e))
        close_camera()

def close_camera():
    global camera, video_writer
    if camera is not None:
        camera.release()
        camera = None
    if video_writer is not None:
        video_writer.release()
        video_writer = None

def generate_frames():
    global is_playing, prediction_list, previous_prediction_time
    frame_delay = 1 / video_fps  # Delay between each frame
    previous_frame_time = time.time()  # Previous frame time
    while is_playing:
        if camera is not None:
            try:
                success, frame = camera.read()
                if not success:
                    break
                else:
                    frame_resized = cv2.resize(frame, (video_width, video_height))

                    # Flip the frame horizontally to fix the mirror effect
                    frame_resized = cv2.flip(frame_resized, 1)

                    # Perform prediction on the frame
                    prediction = predict_frame(frame_resized)

                    # Add prediction text to the frame
                    add_prediction_text_to_frame(frame_resized, prediction)

                    ret, buffer = cv2.imencode('.jpg', frame_resized)
                    frame_encoded = buffer.tobytes()

                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_encoded + b'\r\n')

                    if video_writer is not None:
                        video_writer.write(frame_resized)

                    # Calculate time elapsed for the current frame
                    current_time = time.time()
                    frame_time = current_time - previous_frame_time

                    # Delay to achieve real-time speed
                    if frame_time < frame_delay:
                        time.sleep(frame_delay - frame_time)

                    previous_frame_time = current_time

                    # Calculate average prediction every 60 seconds
                    current_prediction_time = time.time()
                    if current_prediction_time - previous_prediction_time >= 60:
                        average_prediction = calculate_average_prediction(prediction_list)

                        # Save the average prediction to the database
                        with app.app_context():
                            save_prediction_to_database(average_prediction)
                        prediction_list = []
                        previous_prediction_time = current_prediction_time
                    else:
                        prediction_list.append(prediction)

            except Exception as e:
                print(str(e))
                close_camera()
        else:
            break

def calculate_average_prediction(prediction_list):
    # Calculate average prediction based on the received prediction list
    if len(prediction_list) == 0:
        return "No prediction"
    else:
        prediction_counts = {}
        for prediction in prediction_list:
            if prediction in prediction_counts:
                prediction_counts[prediction] += 1
            else:
                prediction_counts[prediction] = 1
        max_count = max(prediction_counts.values())
        average_prediction = [k for k, v in prediction_counts.items() if v == max_count][0]
        return average_prediction

def save_prediction_to_database(average_prediction):
    # Save the prediction to the database
    try:
        prediction = Prediction(prediction=average_prediction)
        db.session.add(prediction)
        db.session.commit()
        print("Prediction saved to database")
    except Exception as e:
        db.session.rollback()
        print("Failed to save prediction to database: {}".format(str(e)))
        
def download_prediction_output():
    return send_file(output_filename, as_attachment=True)

class VideoFeed(Resource):
    def get(self):
        return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

    def post(self):
        global is_playing
        if not is_playing:
            with app.app_context():
                open_camera()
            is_playing = True
        else:
            is_playing = False
            close_camera()
        return {'message': 'Video feed started' if is_playing else 'Video feed stopped'}

    def delete(self):
        global is_playing
        if is_playing:
            is_playing = False
            close_camera()
            average_prediction = calculate_average_prediction(prediction_list)
            save_prediction_to_database(average_prediction)
            return download_prediction_output()
        return {'message': 'No active video feed'}

api.add_resource(VideoFeed, '/predict-video-realtime')

@app.route('/download_output')
def download_output():
    return download_prediction_output()
###########################
##### END: predict video realtime #####
#########################

###########################
##### BEGIN: Register #####
#########################
parserReg = reqparse.RequestParser()
parserReg.add_argument('firstname', type=str,
                       help='firstname', location='json', required=True)
parserReg.add_argument('lastname', type=str,
                       help='lastname', location='json', required=True)
parserReg.add_argument('email', type=str, help='Email',
                       location='json', required=True)
parserReg.add_argument('password', type=str,
                       help='Password', location='json', required=True)
parserReg.add_argument('re_password', type=str,
                       help='Retype Password', location='json', required=True)

@api.route('/register')
class Regis(Resource):
    @api.expect(parserReg)
    def post(self):
        # BEGIN: Get request parameters.
        args = parserReg.parse_args()
        firstname = args['firstname']
        lastname = args['lastname']
        email = args['email']
        password = args['password']
        rePassword = args['re_password']
        is_verified = False
        # END: Get request parameters.

        # BEGIN: Check re_password.
        if password != rePassword:
            return {
                'messege': 'Kata sandi harus sama'
            }, 400
        # END: Check re_password.

        # BEGIN: Check email existance.
        user = db.session.execute(
            db.select(User).filter_by(email=email)).first()
        if user:
            return "Email ini telah digunakan"
        # END: Check email existance.

        # BEGIN: Insert new user.
        user = User()  # Instantiate User object.
        user.firstname = firstname
        user.lastname = lastname
        user.email = email
        user.password = generate_password_hash(password)
        user.is_verified = is_verified
        db.session.add(user)
        msg = Message(subject='Verification OTP', sender=os.environ.get(
            "MAIL_USERNAME"), recipients=[user.email])
        token = random.randrange(10000, 99999)
        session['email'] = user.email
        session['token'] = str(token)
        msg.html = render_template(
            'verify_email.html', token=token)
        mail.send(msg)
        db.session.commit()

        # END: Insert new user.
        return {'messege': 'Registrasi Berhasil, Cek email anda untuk verifikasi'}, 201
#########################
##### END: Register #####
#######################


##############################
##### BEGIN: Verifikasi #####
############################
otpparser = reqparse.RequestParser()
otpparser.add_argument('otp', type=str, help='otp',
                       location='json', required=True)


@api.route('/verifikasi')
class Verifi(Resource):
    @api.expect(otpparser)
    def post(self):
        args = otpparser.parse_args()
        otp = args['otp']
        
        if 'token' in session:
            session_token = session['token']
            if otp == session_token:
                email = session['email']
                user = User.query.filter_by(email=email).first()
                user.is_verified = True
                db.session.commit()
                session.pop('token', None)
                session.pop('email', None)
                return {'message': 'Email berhasil diverifikasi'}, 200
            else:
                return {'message': 'Kode Otp Salah'}, 400
#########################
##### END: Verifikasi #####
#######################

###########################
##### BEGIN: Log in #####
#########################
# import base64
# parserBasic = reqparse.RequestParser()
# parserBasic.add_argument('Authorization', type=str, help='Authorization', location='headers', required=True)


parserLogIn = reqparse.RequestParser()
parserLogIn.add_argument('email', type=str, help='Email',
                         location='json', required=True)
parserLogIn.add_argument('password', type=str,
                         help='Password', location='json', required=True)

SECRET_KEY = "WhatEverYouWant"
ISSUER = "myFlaskWebservice"
AUDIENCE_MOBILE = "myMobileApp"


@api.route('/login')
class Login(Resource):
    @api.expect(parserLogIn)
    def post(self):
        

        # BEGIN: Get request parameters.
        argss = parserLogIn.parse_args()
        email = argss['email']
        password = argss['password']
        # END: Get request parameters.

        if not email or not password:
            return {
                'message': 'Silakan isi email dan kata sandi Anda'
            }, 400

        # BEGIN: Check email existance.
        user = db.session.execute(
            db.select(User).filter_by(email=email)).first()

        if not user:
            return {
                'message': 'Email atau kata sandi salah'
            }, 400
        else:
            user = user[0]  # Unpack the array.
        # END: Check email existance.

        # BEGIN: Check password hash.
        if check_password_hash(user.password, password):
            payload = {
                'user_id': user.id,
                'email': user.email,
                'aud': AUDIENCE_MOBILE,  # AUDIENCE_WEB
                'iss': ISSUER,
                'iat': datetime.utcnow(),
                'exp': datetime.utcnow() + timedelta(hours=8)
            }
            token = jwt.encode(payload, SECRET_KEY)
            return {'message': 'Login Berhasil',
                    'token': token,
                    }, 200
        else:
            return {
                'message': 'Email atau password salah'
            }, 400
        # END: Check password hash.


def decodetoken(jwtToken):
    decode_result = jwt.decode(
        jwtToken,
        SECRET_KEY,
        audience=[AUDIENCE_MOBILE],
        issuer=ISSUER,
        algorithms=['HS256'],
        options={"require": ["aud", "iss", "iat", "exp"]}
    )
    return decode_result
#########################
##### END: Log in #####
#######################


####################################
##### BEGIN: Bearer/Token Auth ####
##################################
authParser = reqparse.RequestParser()
authParser.add_argument('Authorization', type=str,
                        help='Authorization', location='headers', required=True)


@api.route('/detail-user')
class DetailUser(Resource):
    # view user detail
    @api.expect(authParser)
    def get(self):
        args = authParser.parse_args()
        bearerAuth = args['Authorization']
        try:
            jwtToken = bearerAuth[7:]
            token = decodetoken(jwtToken)
            user = db.session.execute(
                db.select(User).filter_by(email=token['email'])).first()
            user = user[0]
            data = {
                'firstname': user.firstname,
                'lastname': user.lastname,
                'email': user.email
            }
        except:
            return {
                'message': 'Token Tidak valid, Silahkan Login Terlebih Dahulu'
            }, 401

        return data, 200
##################################
##### END: Bearer/Token Auth ####
################################


#################################
##### BEGIN: Edit Password #####
###############################
editPasswordParser = reqparse.RequestParser()
editPasswordParser.add_argument(
    'current_password', type=str, help='current_password', location='json', required=True)
editPasswordParser.add_argument(
    'new_password', type=str, help='new_password', location='json', required=True)


@api.route('/edit-password')
class Password(Resource):
    @api.expect(authParser, editPasswordParser)
    def put(self):
        args = editPasswordParser.parse_args()
        argss = authParser.parse_args()
        bearerAuth = argss['Authorization']
        cu_password = args['current_password']
        newpassword = args['new_password']
        try:
            jwtToken = bearerAuth[7:]
            token = decodetoken(jwtToken)
            user = User.query.filter_by(id=token.get('user_id')).first()
            if check_password_hash(user.password, cu_password):
                user.password = generate_password_hash(newpassword)
                db.session.commit()
            else:
                return {'message': 'Password Lama Salah'}, 400
        except:
            return {
                'message': 'Token Tidak valid, Silahkan Login Terlebih Dahulu'
            }, 401
        return {'message': 'Password Berhasil Diubah'}, 200
##################################
##### END: Edit Password ####
################################


#############################
##### BEGIN: Edit user #####
###########################
editParser = reqparse.RequestParser()
editParser.add_argument('firstname', type=str,
                        help='Firstname', location='json', required=True)
editParser.add_argument('lastname', type=str,
                        help='Lastname', location='json', required=True)
editParser.add_argument('Authorization', type=str,
                        help='Authorization', location='headers', required=True)


@api.route('/edit-user')
class EditUser(Resource):
    @api.expect(editParser)
    def put(self):
        args = editParser.parse_args()
        bearerAuth = args['Authorization']
        firstname = args['firstname']
        lastname = args['lastname']
        datenow = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
        try:
            jwtToken = bearerAuth[7:]
            token = decodetoken(jwtToken)
            user = User.query.filter_by(email=token.get('email')).first()
            user.firstname = firstname
            user.lastname = lastname
            user.updatedAt = datenow
            db.session.commit()
        except:
            return {
                'message': 'Token Tidak valid, Silahkan Login Terlebih Dahulu'
            }, 401
        return {'message': 'Update User Berhasil'}, 200
################################
##### END: Edit user ####
################################

if __name__ == '__main__':
    # app.run(debug=True)
    app.run(host='0.0.0.0', port=5000, debug=True)
