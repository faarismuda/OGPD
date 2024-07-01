import subprocess

def application(environ, start_response):
    if 'STREAMLIT_SERVER' not in environ:
        subprocess.Popen(['streamlit', 'run', 'app.py', '--server.port=8080'])
        environ['STREAMLIT_SERVER'] = True

    status = '200 OK'
    output = b'Streamlit app is running...'

    response_headers = [('Content-type', 'text/plain'),
                        ('Content-Length', str(len(output)))]
    start_response(status, response_headers)

    return [output]
