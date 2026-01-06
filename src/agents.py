from crewai import Crew, Agent, Task, Process
from crewai.project import CrewBase, agent, task, crew, before_kickoff, after_kickoff
import json
import os
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_community.utilities.arxiv import ArxivAPIWrapper
from crewai import LLM
from dotenv import load_dotenv
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type

from src.models import PaperRelevance
from src.db_utils import init_db, save_paper_evaluation, get_relevant_summaries

load_dotenv()

# Configure the LLM
llm = LLM(
    model="cerebras/qwen-3-235b-a22b-instruct-2507",
    api_key=os.environ.get("CEREBRAS_API_KEY"),
    base_url="https://api.cerebras.ai/v1",
    temperature=0.5,
)
arxiv_tool = ArxivQueryRun(api_wrapper=ArxivAPIWrapper())

class DBReadTool(BaseTool):
    name: str = "Read Research Database"
    description: str = "Reads the summaries of relevant papers found so far."
    
    def _run(self, query: str = None) -> str:
        # Get summaries from DB
        full_text = get_relevant_summaries("output/research.db")
        
        # Truncate to ~40k characters (approx 10k tokens) to be safe with limits
        max_chars = 40000 
        if len(full_text) > max_chars:
            return full_text[:max_chars] + "\n...[TRUNCATED DUE TO LENGTH]..."
        return full_text

@CrewBase
class ResearchCrew:
    """
    Crew that scours all sources for papers, indexes them, and synthesizes answers.
    """
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"
    
    db_path = "output/research.db"
    paper_output_dir = "output/papers"

    @before_kickoff
    def prepare_inputs(self, inputs):
        """
        Prepare the inputs for the crew
        """
        print("DEBUG: Preparing inputs...")
        # Ensure output directory exists and DB is initialized
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        init_db(self.db_path)
        
        # Normalize inputs
        if isinstance(inputs, str):
            inputs = {"topic": inputs}
        elif inputs is None:
             inputs = {}
             
        if "topic" not in inputs and "question" in inputs:
             inputs["topic"] = inputs["question"]
             
        # Create output dir for papers
        # Note: If called via API, inputs might be dict from start
        out_dir = inputs.get("output_dir", "output")
        self.paper_output_dir = os.path.join(out_dir, "papers")
        self.db_path = os.path.join(out_dir, "research.db")
        
        os.makedirs(self.paper_output_dir, exist_ok=True)
        init_db(self.db_path)
        
        return inputs
    
    @after_kickoff
    def save_inputs(self, result):
        """
        Save the inputs to a file
        """
        return result

    # --- AGENTS ---
    @agent
    def relevance_agent(self):
        return Agent(
            config=self.agents_config["relevance_agent"],
            tools=[arxiv_tool],
            llm=llm,
            verbose=True,
            respect_context_length=False  # Agent 1: Context length not respected (per user req)
        )

    @agent
    def answer_agent(self):
        return Agent(
            config=self.agents_config["answer_agent"],
            llm=llm,
            verbose=True,
            respect_context_length=True   # Agent 2: Respects context
        )

    # --- TASKS ---
    @task
    def search_and_index_task(self):
        def save_paper_callback(output):
            """
            Callback to save the papers to DB and MD files.
            """
            print(f"DEBUG: Callback triggered with type {type(output)}")
            
            # Extract data
            data = None
            if hasattr(output, 'pydantic') and output.pydantic:
                data = output.pydantic
            elif hasattr(output, 'json') and output.json_dict:
                 data = output.json_dict
            
            if not data:
                print("DEBUG: No structured data found in output.")
                return

            # Ensure list
            items = []
            if isinstance(data, list):
                items = data
            elif isinstance(data, PaperRelevance):
                items = [data]
            # Handle container object if Pydantic returned a wrapper
            elif hasattr(data, 'papers'):
                 items = data.papers
            elif isinstance(data, dict):
                 # Try to parse dict if it matches schema
                 try:
                     items = [PaperRelevance(**data)]
                 except:
                     pass

            print(f"DEBUG: Processing {len(items)} items")

            for item in items:
                # Ensure it's a model
                paper = item
                if isinstance(item, dict):
                     try:
                        paper = PaperRelevance(**item)
                     except Exception as e:
                        print(f"Error parsing paper item: {e}")
                        continue
                
                if isinstance(paper, PaperRelevance) and paper.is_relevant:
                    safe_id = str(paper.id).replace('/', '_').replace(':', '_')
                    md_filename = f"{safe_id}.md"
                    md_path = os.path.join(self.paper_output_dir, md_filename)
                    
                    # Save MD
                    with open(md_path, "w", encoding="utf-8") as f:
                        f.write(f"# {paper.id}\n\nRelevance: {paper.relevance_score}\n\n{paper.summary}")
                    
                    # Save to DB
                    save_paper_evaluation(self.db_path, paper, md_path)
                    print(f"Saved relevant paper: {paper.id}")

        return Task(
            config=self.tasks_config["search_and_index_task"],
            agent=self.relevance_agent(),
            output_pydantic=PaperRelevance, # Hint to CrewAI to produce this structure
            callback=save_paper_callback
        )

    @task
    def answer_task(self):
        return Task(
            config=self.tasks_config["answer_task"],
            agent=self.answer_agent()
        ) 

    @crew
    def crew(self):
        """
        Return the crew
        """
        # Inject the DB tool into the answer agent DYNAMICALLY
        # This ensures it reads the DB *after* the search task populates it.
        answer_agent_instance = self.answer_agent()
        answer_agent_instance.tools = [DBReadTool()]

        return Crew(
            agents=[self.relevance_agent(), answer_agent_instance],
            tasks=[self.search_and_index_task(), self.answer_task()],
            process=Process.sequential,
            verbose=True
        )