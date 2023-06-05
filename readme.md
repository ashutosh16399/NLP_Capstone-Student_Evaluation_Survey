# Run Commands-
```
python -m venv myenv
For Windows:
    myenv\Scripts\activate
For Mac:
    source myenv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
celery -A app.celery worker --loglevel=info
```

## on a new terminal Run 
```
For Windows:
    myenv\Scripts\activate
For Mac:
    source myenv/bin/activate
python app.py
```