# How to Run

1. Clone the repository
git clone <repository-url>


2. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate


3. Install dependencies
pip install -r requirements.txt


4. Apply database migrations
python manage.py migrate


5. Create admin user
python manage.py createsuperuser


6. Run Django server (Terminal 1)
python manage.py runserver


7. Run AI worker (Terminal 2)
Open another terminal, activate the same virtual environment, then run:
python ai_worker.py


8. Open in browser
http://127.0.0.1:8000/
