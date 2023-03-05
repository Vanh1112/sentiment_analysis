##Description
## Docker
```
docker build -t tinhte-sentiment:latest .
docker run -dt -v $(pwd):/app/ -p 5000:5000 --env-file docker/.env --name sentiment tinhte-sentiment:latest
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
    "text":"Một nữ hành khách lĩnh án phạt 2 năm tù vì cố gắng mở cửa máy bay lúc đang bay"
    
}
```
response
```
{
  "data": {
    "preprocess_time": 0.0009908676147460938,
    "pred_time": 0.1835179328918457,
    "label": "neutral"
  },
  "elapsed_time": 0.18899297714233398
}
```