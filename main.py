from src.agents import ResearchCrew

def main():
    topic = "models that can replace LLMs, or better than LLMs, or can be the next big model"
    output_dir = "output"
    crew = ResearchCrew()
    crew.kickoff(topic=topic, output_dir=output_dir)

if __name__ == "__main__":
    main()
