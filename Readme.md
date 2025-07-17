# How to Run

Below are step-by-step instructions for **Windows** and **Linux**.

---

## 1. Change into the project folder

```bash
cd Karyotyping_deployment
```

---

## 2. Create a Python virtual environment

### ✅ Windows (CMD or PowerShell)

```cmd
python -m venv venv
```

### ✅ Linux / macOS

```bash
python3 -m venv venv
```

---

## 3. Activate the virtual environment

### ✅ Windows (CMD)

```cmd
venv\Scripts\activate
```

### ✅ Linux / macOS

```bash
source venv/bin/activate
```

---

## 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 5. Run the API with Uvicorn

### ✅ Windows or Linux

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

This will start the FastAPI server at:

```
http://127.0.0.1:8000
```

---

## 🎯 Example Test with CURL

```bash
curl -X POST "http://localhost:8000/karyogram/" ^
 -F "image_file=@<image path>" --output karyo.png
```

Or on Linux/macOS:

```bash
curl -X POST "http://localhost:8000/karyogram/" \
 -F "image_file=@<image path>" --output karyo.png
```

## ✅ Or use swagger

- Swagger UI:
    ```
    http://127.0.0.1:8000/docs
    ```
---

#### The generated karyogram images will be saved in the static/karyograms directory.