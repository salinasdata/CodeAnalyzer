# python base image in the container from Docker Hub
FROM python:3.11.4-slim

# copy files to the /app folder in the container
WORKDIR /usr/src/app

# set the working directory in the container to be /app
COPY . .

# install the packages from the Pipfile in the container
RUN pip install pipenv
RUN pipenv install --system --deploy --ignore-pipfile

# expose the port that uvicorn will run the app on
ENV PORT=8000
EXPOSE 8000
ENV PYTHONPATH "${PYTHONPATH}:/usr/src/app"
# execute the command python main.py (in the WORKDIR) to start the app
CMD ["uvicorn", "run:app", "--host", "0.0.0.0", "--port", "8000"]