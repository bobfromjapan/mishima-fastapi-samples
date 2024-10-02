FROM python:3.10-slim
WORKDIR /code
RUN pip install fastapi==0.109.0 pydantic==2.5.3 uvicorn==0.27.0
COPY ./app.py /code/app.py
EXPOSE 8080
CMD ["python", "app.py"]
