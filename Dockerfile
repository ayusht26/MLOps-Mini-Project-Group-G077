FROM python:3.12-slim

WORKDIR /code

# install libraries first so Docker can cache this layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# then copy the code and the champion model
COPY app ./app
COPY src ./src
COPY models ./models

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
