import os
import sys
from subprocess import Popen

# Tambahkan path aplikasi ke sys.path
sys.path.insert(0, os.path.dirname(__file__))

# Jalankan Streamlit
cmd = ["streamlit", "run", "app.py", "--server.port=8501"]
process = Popen(cmd)

def application(environ, start_response):
    status = '200 OK'
    output = b"Streamlit app is running..."

    response_headers = [('Content-type', 'text/plain'),
                        ('Content-Length', str(len(output)))]
    start_response(status, response_headers)

    return [output]
