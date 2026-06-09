import uvicorn

if __name__ == "__main__":
    # Start the server on localhost:8000 with hot-reloading active
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)