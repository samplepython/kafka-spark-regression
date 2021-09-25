FROM python:3.8

RUN useradd -ms /bin/bash kafkauser
USER kafkauser

WORKDIR /home/kafkauser

COPY ./requirements.txt .

RUN pip install -r requirements.txt && rm requirements.txt

COPY ./src .


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
RUN unset SPARK_HOME && printenv SPARK_HOME || echo "unset success"
RUN echo $SPARK_HOME 
ENV SPARK_HOME=/home/kafkauser/spark-2.4.3-bin-hadoop2.7
RUN echo $SPARK_HOME 
ENV PATH=$PATH:$SPARK_HOME/bin
RUN echo $PATH 
ENV PYSPARK_PYTHON=python3
RUN echo $PYSPARK_PYTHON
RUN pyspark &
RUN ls

ENTRYPOINT spark-submit --jars ./spark-streaming-kafka-0-8-assembly_2.11-2.4.8.jar ./HousePredictionConsumer.py
