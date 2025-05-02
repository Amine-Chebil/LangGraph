from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import httpx
from fastapi.middleware.cors import CORSMiddleware

# Define request and response models
class EmailRequest(BaseModel):
    email_subject: str = ""
    email_text: str

class EmailResponse(BaseModel):
    categories: List[str]
    summary: str

app = FastAPI(title="Hotel Email Classification API")

# Configure CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# LangGraph endpoint
LANGGRAPH_ENDPOINT = "http://localhost:2024/runs/wait"

@app.post("/process-email", response_model=EmailResponse)
async def process_email(request: EmailRequest):
    try:
        # Construct the payload for LangGraph
        payload = {
            "assistant_id": "agent",
            "input": {
                "email_body": request.email_text,
            },
        }
        
        # Call the LangGraph endpoint
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(LANGGRAPH_ENDPOINT, json=payload)
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=500, 
                    detail="Error from LangGraph service"
                )
            
            # Parse the response
            result = response.json()
            
            # Extract data from response
            categories = result.get("predicted_categories", [])
            summary = result.get("email_summary", "")
            
            return EmailResponse(
                categories=categories,
                summary=summary
            )
            
    except httpx.RequestError:
        raise HTTPException(status_code=503, detail="Error connecting to LangGraph service")
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/")
async def root():
    return {"status": "online"}

# Run the API if this file is executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)