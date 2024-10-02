FROM python:3.10
COPY ./app.py /code/app.py
WORKDIR /code
RUN pip install gradio rdkit
EXPOSE 8080
CMD ["python", "app.py"]
