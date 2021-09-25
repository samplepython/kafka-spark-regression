
FROM python:3.6

RUN useradd -ms /bin/bash kafkauser
USER kafkauser

WORKDIR /home/kafkauser

COPY ./requirements.txt .

RUN pip install -r requirements.txt && rm requirements.txt

COPY ./src .

COPY ./ .


RUN pwd
RUN ls
USER root
RUN apt-get update && apt-get upgrade
RUN apt-get install default-jdk -y
USER kafkauser
RUN wget https://archive.apache.org/dist/kafka/2.2.0/kafka_2.12-2.2.0.tgz
RUN tar -xvf kafka_2.12-2.2.0.tgz
RUN mkdir kafka
RUN mv kafka_2.12-2.2.0.tgz kafka
RUN java -version

 
ENTRYPOINT ./Start.sh        
