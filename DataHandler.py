import os
import json
import uuid
import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from flask import Flask

class DataHandler:
    def __init__(self, data_file, base_dir):
        self.data_file = data_file
        self.base_dir = base_dir
        self.data = self.load_data()

    def load_data(self):
        if os.path.exists(self.data_file):
            with open(self.data_file, 'r',encoding="utf8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = {}
        else:
            data = {}

        return data

    def save_data(self):
        with open(self.data_file, "w", encoding="utf8") as f:
            json.dump(self.data, f, indent=4)

    def add_csv_file(self, email_id, file):
        filename = file.filename
        if email_id not in self.data:
            self.data[email_id] = {"csv_files": {}}

        # Generate a unique CSV ID using UUID
        csv_id = str(uuid.uuid4())

        # Generate a unique filename using CSV ID
        unique_filename = csv_id

        # Create the email directory if it doesn't exist
        email_dir = os.path.join(self.base_dir, email_id)
        if not os.path.exists(email_dir):
            os.makedirs(email_dir)

        # Save the uploaded CSV file to the email directory
        file_path = os.path.join(email_dir, unique_filename + ".csv")
        file.save(file_path)

        # Save the CSV ID and file information in the data
        self.data[email_id]["csv_files"][csv_id] = {
            "filename": unique_filename,
            "created_at": str(datetime.datetime.now())
        }

        self.save_data()
        return csv_id

    def remove_csv_file(self, email_id, csv_id):
        if email_id in self.data and csv_id in self.data[email_id]["csv_files"]:
            filename = self.data[email_id]["csv_files"][csv_id]["filename"]
            file_path = os.path.join(self.base_dir, email_id, filename)

            # Remove the file if it exists
            if os.path.exists(file_path):
                os.remove(file_path)

            del self.data[email_id]["csv_files"][csv_id]

            # Remove the email if there are no more CSV files associated with it
            if not self.data[email_id]["csv_files"]:
                self.remove_email(email_id)

            self.save_data()

    def remove_email(self, email_id):
        if email_id in self.data:
            # Remove the email directory from the base directory
            email_dir = os.path.join(self.base_dir, email_id)
            if os.path.exists(email_dir):
                for filename in os.listdir(email_dir):
                    file_path = os.path.join(email_dir, filename)
                    os.remove(file_path)
                os.rmdir(email_dir)

            del self.data[email_id]
            self.save_data()

    def get_csv_files(self, email_id):
        if email_id in self.data:
            return self.data[email_id]["csv_files"]
        else:
            return {}

    def cleanup_csv_files(self, max_age_days):
        current_datetime = datetime.datetime.now()

        for email_id in self.data.keys():
            csv_files = self.get_csv_files(email_id)
            csv_ids_to_delete = []

            for csv_id, csv_info in csv_files.items():
                created_at = datetime.datetime.strptime(csv_info["created_at"], "%Y-%m-%d %H:%M:%S.%f")
                age = current_datetime - created_at

                if age.days >= max_age_days:
                    csv_ids_to_delete.append(csv_id)

            for csv_id in csv_ids_to_delete:
                self.remove_csv_file(email_id, csv_id)

    def get_email_from_csv_id(self, csv_id):
        for email_id, email_data in self.data.items():
            csv_files = email_data.get("csv_files", {})
            if csv_id in csv_files:
                return email_id
        return None