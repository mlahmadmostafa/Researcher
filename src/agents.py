from crewai import Crew, Agent, Task, Process
from crewai.project import CrewBase, agent, task, crew, before_kickoff, after_kickoff
import json
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_community.utilities.arxiv import ArxivAPIWrapper
from crewai import LLM
from dotenv import load_dotenv
import os

load_dotenv()
# Configure the LLM to use Cerebras
llm = LLM(
    model="cerebras/qwen-3-235b-a22b-instruct-2507", # Replace with your chosen Cerebras model name, e.g., "cerebras/llama3.1-8b"
    api_key=os.environ.get("CEREBRAS_API_KEY"), # Your Cerebras API key
    base_url="https://api.cerebras.ai/v1",
    temperature=0.5,
    # Optional parameters:
    # top_p=1,
    # max_completion_tokens=8192, # Max tokens for the response
    # response_format={"type": "json_object"} # Ensures the response is in JSON format
)
arxiv_tool = ArxivQueryRun(api_wrapper=ArxivAPIWrapper())
@CrewBase
class ResearchCrew:
    """
    Crew that scoures all sources for papers and embeds them then creates a massive reports on said topics
    """
    agents_config ="config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @before_kickoff
    def prepare_inputs(self, topic: str, output_dir: str):
        """
        Prepare the inputs for the crew
        """
        self.inputs = {
            "topic": self.topic,
            "output_dir": self.output_dir,
        }
    
    @after_kickoff
    def save_inputs(self):
        """
        Save the inputs to a file
        """
        with open("inputs.json", "w") as f:
            json.dump(self.inputs, f)
    # Looking for papers on the topic
    @agent
    def search_papers(self):
        """
        Search for papers on the topic
        """
        return Agent(
            config = self.agents_config["search_papers"],
            tools = [arxiv_tool],
            llm = llm,
            respect_context_length = False,
        )
    @task
    def search_papers_task(self):
        """
        Search for papers on the topic
        """
        return Task(
            config = self.tasks_config["search_papers"],
        ) 
    # Investigating papers and creating a report
    @crew
    def crew(self):
        """
        Return the crew
        """
        return Crew(
            agents = [self.search_papers()],
            tasks = [self.search_papers_task()],
            process=Process.sequential,
            verbose=True,
            embedder={
                "provider": "ollama",
                "config": {
                    "model": "embeddinggemma:300m-qat-q4_0",  # or "nomic-embed-text"
                    "url": "http://localhost:11434/api/embeddings"  # Default Ollama URL
                }
            }
        )
        