from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/kickoff")
def kickoff():
    return {"question": "Crew kickoff"}

@