# Run Commands-
```
python -m venv myenv
myenv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
celery -A app.celery worker --loglevel=info
```

## on a new terminal Run 
```
myenv\Scripts\activate
python app.py
```