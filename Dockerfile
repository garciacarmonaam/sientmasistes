FROM python:3.7-slim
COPY . /main
WORKDIR /main
RUN pip install -r requirements.txt
EXPOSE 80
RUN mkdir ~/.streamlit
RUN cp config.toml ~/.streamlit/config.toml
RUN cp credentials.toml ~/.streamlit/credentials.toml
WORKDIR /main
ENTRYPOINT ["streamlit", "run"]
CMD ["main.py"]