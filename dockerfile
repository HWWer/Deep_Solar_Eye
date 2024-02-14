FROM python:3.10

WORKDIR /DeepSolarEye/api

COPY api_requirements.txt /DeepSolarEye/api/
RUN pip install --no-cache-dir -r api_requirements.txt

COPY . /DeepSolarEye/api/

CMD uvicorn DeepSolarEye.api.fast_api:app --host 0.0.0.0 --port 8080
