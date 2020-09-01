FROM python:3

# Create app directory
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

# Install app dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Bundle app source
COPY elbotto ./elbotto
COPY setup.py README.rst MANIFEST.in HISTORY.rst runUnix.sh ./
RUN pip install .

EXPOSE 6006
#ENTRYPOINT [ "python", "./elbotto/launcher.py" ]
ENTRYPOINT [ "./runUnix.sh" ]
