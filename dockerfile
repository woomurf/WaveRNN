FROM pytorch/pytorch:1.5.1-cuda10.1-cudnn7-runtime

COPY . . 

RUN pip install -r requirements.txt
RUN pip install Flask
RUN pip install -U flask-cors
RUN pip install numba==0.48

EXPOSE 80 

ENTRYPOINT ["python"]
CMD ["server.py"]