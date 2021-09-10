FROM python:3.8

RUN useradd -ms /bin/bash kafkauser
USER kafkauser

WORKDIR /home/kafkauser

COPY Kafka-Producer/requirements.txt .

RUN pip install -r requirements.txt && rm requirements.txt

COPY Kafka-Producer/src .


RUN pwd
RUN ls
RUN wget https://archive.apache.org/dist/kafka/2.2.0/kafka_2.12-2.2.0.tgz
RUN tar -xvf kafka_2.12-2.2.0.tgz
RUN mkdir kafka
RUN mv kafka_2.12-2.2.0.tgz kafka
USER root
RUN echo "deb http://security.debian.org/debian-security stretch/updates main" >> /etc/apt/sources.list
RUN apt-get update -y  
RUN wget http://security.ubuntu.com/ubuntu/pool/universe/o/openjdk-8/openjdk-8-jdk-headless_8u292-b10-0ubuntu1~18.04_amd64.deb
RUN wget http://security.ubuntu.com/ubuntu/pool/universe/o/openjdk-8/openjdk-8-jdk_8u292-b10-0ubuntu1~18.04_amd64.deb
RUN wget http://security.ubuntu.com/ubuntu/pool/universe/o/openjdk-8/openjdk-8-jre-headless_8u292-b10-0ubuntu1~18.04_amd64.deb
RUN wget http://security.ubuntu.com/ubuntu/pool/universe/o/openjdk-8/openjdk-8-jre_8u292-b10-0ubuntu1~18.04_amd64.deb
RUN dpkg -i openjdk-8-jre_8u292-b10-0ubuntu1~18.04_amd64.deb openjdk-8-jre-headless_8u292-b10-0ubuntu1~18.04_amd64.deb  openjdk-8-jdk_8u292-b10-0ubuntu1~18.04_amd64.deb openjdk-8-jdk-headless_8u292-b10-0ubuntu1~18.04_amd64.deb
USER kafkauser
RUN java -version
RUN pip3 install kakfa-python
RUN pip3 list | grep kafka
RUN cd kafka_2.12-2.2.0/
RUN bin/zookeeper-server-start.sh config/zookeeper.properties & 
RUN sleep 30
RUN bin/kafka-server-start.sh config/server.properties &
RUN sleep 30
RUN bin/kafka-topics.sh --list --bootstrap-server localhost:9092
RUN bin/kafka-topics.sh --create --bootstrap-server localhost:9092 --replication-factor 1 --partitions 1 --topic house-price-prediction
RUN bin/kafka-topics.sh --list --bootstrap-server localhost:9092
RUN ls
RUN cd ..


RUN python /home/kafkauser/Kafka-Producer/src/TestDataStreamProducer.py

