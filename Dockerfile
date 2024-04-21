# Base image olarak resmi Python imajını kullanın
FROM python:3.9

# Çalışma dizinini ayarlayın
WORKDIR /app

# Gerekli Python paketlerini yükleyin
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Uygulama kodunuzu kopyalayın
COPY . .

# Uygulamanızın çalıştırılacağı portu belirtin
EXPOSE 8000

# Uygulamayı başlatma komutu
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
