from flask import Flask, abort, jsonify, request, render_template
import os
from dotenv import load_dotenv
from pathlib import Path, PosixPath
import csv
import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from celery import Celery
from flask_mail import Mail, Message
from DataHandler import DataHandler
import pandas as pd
import requests
import model


def initialise_data_handler(base_path, data_file):
    # Check if data file exists
    fpath = os.path.join(base_path, data_file)
    if not os.path.isfile(fpath):
        # Create an empty file if it doesn't exist
        with open(fpath, 'w',encoding="utf8") as f:
            pass
    # Create an instance of the DataHandler class
    dh = DataHandler(fpath, base_path)
    return dh


BASE_PATH = os.path.join(os.path.dirname(__file__), "uploads")
DATA_HANDLER_FILE = 'data.json'
dh = initialise_data_handler(BASE_PATH, DATA_HANDLER_FILE)


def delete_old_csv_files():
    max_age_days = 10  # Maximum age of CSV files in days
    dh.cleanup_csv_files(max_age_days)


scheduler = BackgroundScheduler()
scheduler.add_job(delete_old_csv_files, 'interval', days=1)  # Run the function every day
scheduler.start()

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config['MAILGUN_API_KEY'] = 'ddca89a09e1673f342bf4d91c74a70f2-07ec2ba2-97026b86'
app.config['MAILGUN_DOMAIN'] = 'sandbox2e77a264437341cc910652b502dbf4ea.mailgun.org'


celery = Celery(app.name, broker='redis://localhost:6379/0')
celery.conf.update(app.config)
mail = Mail(app)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/email", methods=["GET"])
def email():
    return render_template("email.html")


@app.route("/process", methods=["POST"])
def process():
    email = request.form.get('email')
    file = request.files["file"]
    if "file" not in request.files:
        return abort(400, "No file attached")
    if not email:
        return jsonify({"message": "Email input is missing"}), 400
    if not file:
        return abort(400, "No file attached")
    if not file.filename.endswith(".csv"):
        return jsonify({"message": "The file is not a CSV file"}), 400
    file_name = file.filename
    file_id = dh.add_csv_file(email, file)

    return jsonify({"message": "File uploaded successfully", "file_id": file_id, "email": email}), 200


@app.route("/<file_id>/uploads", methods=["GET"])
def view_data(file_id):
    csv_id = file_id
    email = dh.get_email_from_csv_id(csv_id)
    if email:
        print("Email ID:", email)
    else:
        print("Email not found for the given CSV ID.")
    csv_file = os.path.join(BASE_PATH, email, f"{file_id}.csv")
    if not os.path.exists(csv_file):
        abort(404, "File not found")

    with open(csv_file, "r", encoding="utf8") as f:
        reader = csv.reader(f)
        return render_template("data.html", csv=reader)

@app.route("/result/<file_id>", methods=["GET"])
def display_result(file_id):
    email = dh.get_email_from_csv_id(file_id)
    if not email:
        abort(404, "Email not found for the given CSV ID.")

    csv_file = os.path.join(BASE_PATH, email, f"{file_id}.csv")
    if not os.path.exists(csv_file):
        abort(404, "File not found")

    with open(csv_file, "r",encoding="utf8") as f:
        reader = csv.reader(f)
        csv_data = list(reader)

    return render_template("result.html", csv_data=csv_data)


@celery.task
def long_running_task(email, csv_id):
    file_path = os.path.join(BASE_PATH, email, csv_id + '.csv')
    print("Starting task...")
    print("File path:", file_path)

    try:
        df = pd.read_csv(file_path)  # Specify the engine as 'openpyxl'
        file_s = open(r"Stopwords.csv", "r")
        Stop_data = list(csv.reader(file_s, delimiter=","))
        file_s.close()
        df = model.call_everything(df,Stop_data)
        output_file_path = os.path.join(BASE_PATH, email, csv_id + '.csv')
        df.to_csv(output_file_path, index=False)

        # ...
        print("Task completed successfully!")
    except Exception as e:
        print("An error occurred:", str(e))
        return

    url = 'http://127.0.0.1:5000/result/' + csv_id
    print(url)
    send_email(email, url)


def send_email(email, url):
    print('the send to : ' + email)
    api_key = app.config['MAILGUN_API_KEY']
    domain = app.config['MAILGUN_DOMAIN']
    mailgun_url = f"https://api.mailgun.net/v3/{domain}/messages"

    auth = ('api', api_key)
    data = {
        'from': 'Your Name <postmaster@sandbox2e77a264437341cc910652b502dbf4ea.mailgun.org>',
        'to': email,
        'subject': 'Task Complete',
        'text': f'Your task is complete. Please check the results at URL: {url}'
    }
    response = requests.post(mailgun_url, auth=auth, data=data)

    if response.status_code == 200:
        print('Email sent successfully')
    else:
        print('Failed to send email')


@app.route('/perform_model_task', methods=['POST'])
def perform_task():
    data = request.get_json()
    csv_id = data.get('csv')
    print('\nCSV ID' + str(csv_id))
    email = dh.get_email_from_csv_id(csv_id)
    if email:
        print("Email ID:", email)
    else:
        print("Email not found for the given CSV ID.")
    if not email:
        return jsonify({'message': 'Email input is missing'}), 400
    if not csv_id:
        return jsonify({'message': 'CSV input is missing'}), 400

    long_running_task.delay(email, csv_id)  # Enqueue the task for asynchronous execution

    return jsonify({'message': 'Task enqueued'})


@app.errorhandler(404)
def page_not_found(e):
    return render_template("error.html", error=e), 404


@app.errorhandler(400)
def bad_request(e):
    return render_template("error.html", error=e), 400


@app.errorhandler(500)
def server_error(e):
    return render_template("error.html", error=e), 500


if __name__ == "__main__":
    app.run(debug=True)
    app.config["TEMPLATES_AUTO_RELOAD"] = True

