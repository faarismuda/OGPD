#!/bin/bash
source /home/ogpdmyid/virtualenv/public_html/OGPD/3.11/bin/activate
cd /home/ogpdmyid/public_html/OGPD
nohup streamlit run app.py --server.port=8501 &
