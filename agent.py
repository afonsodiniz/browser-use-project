import logging
from langchain_anthropic import ChatAnthropic
from browser_use import Agent
import asyncio
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# You can adjust the level to control verbosity: DEBUG, INFO, WARNING, ERROR, CRITICAL
# For less verbose output, change DEBUG to INFO

load_dotenv()

async def main():
    agent = Agent(
        task="""
                Go to https://www.continente.pt/login/ and add user @gmail.com and password $.

        
                To add to the cart each item, you have to click on 'Carrinho', and specify the number of items as stated below:                
                - Pão de Centeio Serra da Estrela Fatiado: 1 un
                - Feijão Verde: 1 un
                - Tomate Cherry Trimix: 1 un
                - Bifes de Frango: 1 un
                - Pepino: 2 un
                - Tomate Chucha: 4 un
                - Banana: 6 un
                - Clementina: 5 un
                - Cenoura: 4 un
                - Batata Doce: 3 un
                - Maçã Pink Lady: 6 un
                - Iogurte Skyr Mirtilo: 3 un
                - Iogurte Grego Morango: 1 un
                - Ovos de Solo Classe L: 1 un
                - Iogurte Skyr Morango: 3 un
                - Arroz Basmati: 1 un
                - Massa Esparguete Pack Poupança: 1 un
                - Massa Espirais: 1 un
                - Doce 4 Frutos: 1 un
                
                After adding all items, click on the cart icon on the top right corner, and print the total cost of the cart.
                """,
        llm=ChatAnthropic(model="claude-3-5-sonnet-20241022"),
    )
    history = await agent.run()
    
    # Print token usage statistics
    total_input_tokens = history.total_input_tokens()
    print(f"Total input tokens used: {total_input_tokens}")
    
    # Get token usage per step
    tokens_per_step = history.input_token_usage()
    print(f"Tokens used per step: {tokens_per_step}")
    
    # Calculate approximate cost (for Claude-3.5-sonnet)
    input_cost = total_input_tokens / 1_000_000 * 3  # $3 per 1M tokens
    # Note: Output tokens are harder to track directly
    print(f"Approximate input cost: ${input_cost:.4f}")
    
    # Access the message manager's token counts
    message_manager = agent.message_manager
    current_tokens = message_manager.state.history.current_tokens
    print(f"Current tokens in context: {current_tokens}")
    
    # Print tokens for each message
    for i, msg in enumerate(message_manager.state.history.messages):
        print(f"Message {i}: {msg.metadata.tokens} tokens")

asyncio.run(main())
