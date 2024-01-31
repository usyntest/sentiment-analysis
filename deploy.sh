cd /home/ubuntu/sentiment-analysis/

# Create virtual environment if does not exist already.
if [ ! -d "$DIRECTORY" ]; then
        virtualenv venv
        source venv/bin/activate
        pip install -r requirements.txt
        python3 -c "import nltk; nltk.download('popular');"
fi

source venv/bin/activate

gunicorn -w 4 -b 127.0.0.1:54321 'server:app'
