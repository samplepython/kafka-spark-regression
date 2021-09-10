FROM python:3.8

RUN useradd -ms /bin/bash kafkauser
USER kafkauser

WORKDIR /home/kafkauser

COPY Kafka-Consumer/requirements.txt .

RUN pip install -r requirements.txt && rm requirements.txt

COPY Kafka-Consumer/src .


RUN pwd
RUN ls
RUN wget https://archive.apache.org/dist/spark/spark-2.4.3/spark-2.4.3-bin-hadoop2.7.tgz
RUN tar -xvf spark-2.4.3-bin-hadoop2.7.tgz
RUN mkdir spark
RUN mv spark-2.4.3-bin-hadoop2.7.tgz spark
USER root
RUN apt-get update -y && apt-get install scala -y
USER kafkauser
RUN scala -version
RUN pip3 install pyspark==2.4.6
RUN pip3 list | grep spark
RUN echo "export PATH=$PATH:/home/kafkauser/spark-2.4.3-bin-hadoop2.7/bin" >> ~/.bashrc
RUN echo "export PYSPARK_PYTHON=python3" >> ~/.bashrc
RUN . ~/.bashrc
RUN echo $PATH
RUN pyspark
RUN exit
RUN ls


RUN python spark-submit --jars /home/kafkauser/Kafka-Consumer/src/spark-streaming-kafka-0-8-assembly_2.11-2.4.8.jar /home/kafkauser/Kafka-Consumer/src/HousePredictionConsumer.py
