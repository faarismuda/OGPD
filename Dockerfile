# Gunakan base image yang telah terinstall Python
FROM python:3.11

# Set working directory
WORKDIR /app

# Instal curl untuk mengunduh Node.js
RUN apt-get update && apt-get install -y curl gnupg

# Instal Node.js 16.20.2
RUN curl -fsSL https://deb.nodesource.com/setup_16.x | bash - \
    && apt-get install -y nodejs=16.20.2-1nodesource1

# Salin semua file proyek ke dalam container
COPY . .

# Instal dependencies Python
RUN pip install -r requirements.txt

# Instal @playwright/test
RUN npm install -g @playwright/test

# Instal dependencies Playwright
RUN npx playwright install-deps

# Instal Microsoft Edge dari file instalasi lokal
RUN apt-get update && apt-get install -y /app/assets/microsoft-edge-stable_126.0.2592.81-1_amd64.deb \
    && rm /app/assets/microsoft-edge-stable_126.0.2592.81-1_amd64.deb

# Pastikan msedgedriver memiliki izin eksekusi dan pindahkan ke /usr/local/bin
RUN chmod +x /app/assets/msedgedriver && mv /app/assets/msedgedriver /usr/local/bin/msedgedriver

# Eksekusi streamlit saat container berjalan
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
