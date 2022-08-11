FROM python:3.10-slim-buster
#copy requirements.txt file into image
COPY ./requirements.txt /app/requirements.txt

#switch working directory

WORKDIR /app

#intsall dependencies

RUN pip install -r requirements.txt

#copy every content from the local file to images

COPY . /app


#configure the container to run in an executed manner

ENTRYPOINT [ "python" ]

CMD [ "app.py"]



