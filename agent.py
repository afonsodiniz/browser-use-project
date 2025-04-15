import logging
from langchain_anthropic import ChatAnthropic
from browser_use import Agent, Browser
import asyncio
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

load_dotenv()

task = """
    ### Prompt for Shopping Agent – Continente Online Grocery Order

**Objective:** Login to Continente online, add specific items with exact quantities to the cart, and review the total cost.

**Important:**
- Add exactly the specified quantities for each item
- If an item is unavailable, look for a similar alternative
- Follow the exact steps in order

---

### Step 1: Login to the Website
- Go to https://www.continente.pt/login/
- Enter username: 
- Enter password: 
- Click the login button

---

### Step 2: Add Items to Cart

**Bakery:**
- Pão de Centeio Serra da Estrela Fatiado: 1 un

**Vegetables:**
- Feijão Verde: 1 un
- Tomate Cherry Trimix: 1 un
- Pepino: 2 un
- Tomate Chucha: 4 un
- Cenoura: 4 un

**Fruits:**
- Banana: 6 un
- Clementina: 5 un
- Maçã Pink Lady: 6 un

**Meat:**
- Bifes de Frango: 1 un

**Root Vegetables:**
- Batata Doce: 3 un

**Dairy & Eggs:**
- Iogurte Skyr Mirtilo: 3 un
- Iogurte Grego Morango: 1 un
- Iogurte Skyr Morango: 3 un
- Ovos de Solo Classe L: 1 un

**Pantry Items:**
- Arroz Basmati: 1 un
- Massa Esparguete Pack Poupança: 1 un
- Massa Espirais: 1 un
- Doce 4 Frutos: 1 un

---

### Step 3: Handling Item Search and Addition
- For each item, search using the search bar if needed
- Click on the 'Carrinho' button next to each item
- Adjust the quantity to match exactly what is specified
- Make sure the item is successfully added before moving to the next

---

### Step 4: Review Cart and Total
- After adding all items, click on the cart icon in the top right corner
- Review all items to ensure they match the specified quantities
- Note the total cost of all items
- Print or report the total cost

---

### Step 5: Output Summary
- Provide a detailed list of all items successfully added to cart
- Note any items that were unavailable or substituted
- Report the final cart total
"""

browser = Browser()
agent = Agent(
    task=task,
    llm=ChatAnthropic(model="claude-3-5-sonnet-20241022"),
    browser=browser,
)

async def main():
    history = await agent.run()
    
    # Print token usage statistics
    total_input_tokens = history.total_input_tokens()
    print(f"Total input tokens used: {total_input_tokens}")
    
    # Get token usage per step
    tokens_per_step = history.input_token_usage()
    print(f"Tokens used per step: {tokens_per_step}")
    
    # Calculate approximate cost (for Claude-3.5-sonnet)
    input_cost = total_input_tokens / 1_000_000 * 3  # $3 per 1M tokens
    print(f"Approximate input cost: ${input_cost:.4f}")
    
    # Access the message manager's token counts
    message_manager = agent.message_manager
    current_tokens = message_manager.state.history.current_tokens
    print(f"Current tokens in context: {current_tokens}")
    
    # Print tokens for each message
    for i, msg in enumerate(message_manager.state.history.messages):
        print(f"Message {i}: {msg.metadata.tokens} tokens")
    
    input('Press Enter to close the browser...')
    await browser.close()

if __name__ == '__main__':
    asyncio.run(main())