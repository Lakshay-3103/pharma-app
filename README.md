💊 Medicine Price & Alternative Finder
  A full-stack machine learning application designed to predict fair pharmaceutical prices and recommend cheaper manufacturer alternatives built using FastAPI Backend and      streamlit Frontend.

🚀 Live Demo
      Frontend UI: pharma-app-a2djh7gfyp92dgotydhvnm.streamlit.app
      Backend API: https://medicine-api-48l2.onrender.com

📂 Project Structure
  Following a clean separation of concerns, the repository is structured as follows:
  pharma-app/
  │
  ├── backend/       # FastAPI application and routing
  ├── frontend/      # Streamlit user interface
  ├── model/         # Serialized ML model (compressed with joblib)
  ├── data/          # Training datasets and processing scripts
  ├── requirements.txt
  └── README.md


🧠 Model Performance & Data Processing

 Composition Cleaning Strategy:
 To avoid the common pitfall of exact string matching (e.g., missing the connection between "Paracetamol 500mg" and "paracetamol(500 mg)"), the data pipeline implements a   robust cleaning step before alternative matching. Text is standardized by converting to lowercase, stripping special characters and parentheses, and normalizing            whitespace,  ensuring a highly accurate alternative retrieval system.

 Model Accuracy:
 The Random Forest regressor evaluates to a Mean Absolute Error (MAE) of ₹182.13. The model was trained offline, with only the compressed .joblib artifact loaded into the   FastAPI backend during runtime to optimize API response times.


🏗️ System Architecture

Frontend: Built with Streamlit for a responsive, interactive UI. Hosted on Streamlit Community Cloud.

Backend: RESTful API built with FastAPI. Hosted on Render.

Machine Learning: Scikit-Learn (Random Forest) and Pandas. Model artifact is optimized using joblib compression (compress=5) to maintain a lightweight footprint and bypass GitHub file size limits.


💻 How to Run Locally (Bash Code)
 1. Clone the repository:
    git clone https://github.com/Lakshay-3103/pharma-app.git
    cd pharma-app

2. Install dependencies:
    pip install -r requirements.txt

3. Start the FastAPI Backend:
    cd backend
    uvicorn app:app --reload --port 10000

4. Start the Streamlit Frontend (in a new terminal):
    cd frontend
    streamlit run app.py
  
