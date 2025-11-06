from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI(title="Compliment Generator for n8n")

# Allow CORS for all origins (useful for n8n cloud)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Groq LLM
llm = ChatGroq(
    model="gemma-2-9b-it",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.7
)

# ---- Models ----
class InputData(BaseModel):
    data: dict  # expects { "description": "..." }

class OutputData(BaseModel):
    text: str   # n8n-compatible output


# ---- Routes ----
@app.get("/")
def home():
    return {"message": "Custom Compliment Generator API is running!"}


@app.post("/generate", response_model=OutputData)
def generate_compliment(request: InputData):
    try:
        description = request.data.get("description", "").strip()
        if not description:
            raise HTTPException(status_code=400, detail="Missing 'description' in data field")

        # System prompt as per your specification
        system_prompt = (
            "You're crafting very short, natural-sounding compliments or congratulations "
            "for cold outreach emails based on the company's website content.\n\n"
            "Instructions: First priority: Look carefully for specific recent events, awards, "
            "achievements, milestones, events, notable numbers, or news mentioned on their website. "
            "If nothing specific is found, give a simple, believable compliment based on something "
            "that seems authentic or genuinely noteworthy about the company's approach.\n\n"
            "Rules:\n"
            "- Keep it under 100 words\n"
            "- Sound natural like a student is writing and conversational\n"
            "- Be specific and authentic\n"
            "- Avoid being too formal or overly enthusiastic"
        )

        # Combine system + user context
        user_prompt = f"Website content: {description}"

        # Get AI response
        response = llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])

        # Return in n8n-compatible format
        return OutputData(text=response.content.strip())

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating compliment: {str(e)}")


# ---- For Railway Deployment ----
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
