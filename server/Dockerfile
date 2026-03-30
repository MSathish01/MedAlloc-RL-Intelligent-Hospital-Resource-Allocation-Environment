FROM python:3.10

WORKDIR /app

COPY . .

RUN pip install fastapi uvicorn

EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]