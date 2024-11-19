# OCR_Dates




## Building docker image
docker build -t my-app .


## Requirements

1) Python 3.11 or later.

2) Build the Docker Image::
```bash
docker build -t my-app .
docker run -p 8080:8080 my-app
```
3) Run the Container:
```bash
docker run -p 8080:8080 my-app
```

## Access the App:
* Open http://localhost:5000 in your browser.
* The webpage served from templates/ is displayed.
* Users can upload an image, which is processed using the OCR model.


.
├── main.py          # Python script for your Flask app
├── model.h5         # Pre-trained model for OCR predictions
├── templates/       # Folder containing HTML files for the web UI
│   └── index.html
└── Dockerfile       # Dockerfile for containerizing the application
