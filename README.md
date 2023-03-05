##Description
## Docker
```
docker build -t sentiment_analysis:latest .
docker run -dt -v $(pwd):/app/ -p 5000:5000 --env-file docker/.env --name sentiment sentiment_analysis:latest
```

## Guide
```
Train: python3 -m app.train
App: python3 -m app.run
```
## API
request
```
POST
{
    "text":"Cũng được nhưng mà không ra gì"
    
}
```
response
```
{
  "data": {
    "preprocess_time": 0.0009908676147460938,
    "pred_time": 0.1835179328918457,
    "label": "negative"
  },
  "elapsed_time": 0.18899297714233398
}
```
