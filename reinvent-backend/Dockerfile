FROM python:3.10
COPY ./REINVENT4 /code/reinvent
WORKDIR /code/reinvent
RUN pip install -r requirements-linux-64.lock
RUN pip install --no-deps .
RUN pip install fastapi==0.99.1 uvicorn==0.27.0 pydantic==1.10.5
COPY ./app.py /code/reinvent/app.py
EXPOSE 8080
CMD ["python", "app.py"]
