from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import httpx
import json
from groq import Groq
import re
from enum import Enum
import time
from decimal import Decimal, InvalidOperation
from dotenv import load_dotenv
import os

# Load the .env file
load_dotenv()


app = FastAPI(title="1inch AI-Driven Smart Vault", version="1.0.0")


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ONE_INCH_API_KEY = os.getenv("ONE_INCH_API_KEY")


# Initialize Groq client
groq_client = Groq(api_key=GROQ_API_KEY)


# --- Enums for clarity ---
class StrategyType(str, Enum):
   LIMIT_ORDER = "limit_order"
   MARKET_MAKING = "market_making"


class Intent(str, Enum):
   BUY = "buy"
   SELL = "sell"
   PROVIDE_LIQUIDITY = "provide_liquidity"
   SWAP = "swap"


class TriggerType(str, Enum):
   PRICE_BASED = "price_based"
   TIME_BASED = "time_based"
   IMMEDIATE = "immediate"
   PERCENTAGE_BASED = "percentage_based"


class ActionType(str, Enum):
   CREATE_ORDER = "create_order"
   CANCEL_ORDER = "cancel_order"
   UPDATE_ORDER = "update_order"


# --- Request/Response Models ---
class UserQuery(BaseModel):
    query: str
    wallet_address: str
    receiver_address: Optional[str] = None # User can explicitly provide this


class ClassifiedIntent(BaseModel):
   strategy: StrategyType
   intent: Intent
   trigger: TriggerType
   action: ActionType
   asset_from: str
   asset_to: str
   amount: Optional[str] = None
   price_target: Optional[str] = None
   trigger_value: Optional[str] = None
   confidence_score: float = Field(..., ge=0.0, le=1.0) # Ensure score is between 0 and 1
   receiver_address: Optional[str] = None # Extracted from query or provided directly


class APIResponse(BaseModel):
   success: bool
   intent: Optional[ClassifiedIntent] = None
   api_endpoint: Optional[str] = None
   api_method: Optional[str] = None
   api_parameters: Optional[Dict[str, Any]] = None
   error_message: Optional[str] = None
   guidance: Optional[str] = None


# ---
# ## Token Information Service
# This service dynamically fetches token data from 1inch or uses a hardcoded fallback.
# ---
class TokenInfo:
   def __init__(self, symbol: str, address: str, decimals: int):
       self.symbol = symbol
       self.address = address
       self.decimals = decimals


class TokenInfoService:
   _token_cache: Dict[str, TokenInfo] = {} # Cache token info
   ETHEREUM_CHAIN_ID = 1 # Common chain for stablecoins and major assets


   # Hardcoded fallback for common tokens - in case 1inch API fails or for very popular assets
   # These decimals are standard for these tokens on Ethereum.
   _hardcoded_tokens: Dict[str, TokenInfo] = {
       "USDT": TokenInfo("USDT", "0xdAC17F958D2ee523a2206206994597C13D831ec7", 6),
       "USDC": TokenInfo("USDC", "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48", 6),
       "ETH": TokenInfo("ETH", "0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE", 18), # Special address for native ETH
       "WETH": TokenInfo("WETH", "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2", 18),
       "DAI": TokenInfo("DAI", "0x6B175474E89094C44Da98b954EedeAC495271d0F", 18),
       "WBTC": TokenInfo("WBTC", "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599", 8), # WBTC has 8 decimals
   }


   @classmethod
   async def get_token_info(cls, symbol: str) -> Optional[TokenInfo]:
       symbol_upper = symbol.upper()


       if symbol_upper in cls._token_cache:
           return cls._token_cache[symbol_upper]


       # 1. Try hardcoded tokens first (fastest fallback)
       if symbol_upper in cls._hardcoded_tokens:
           cls._token_cache[symbol_upper] = cls._hardcoded_tokens[symbol_upper]
           return cls._hardcoded_tokens[symbol_upper]


       # 2. Try 1inch's public /token API (now with Authorization header)
       try:
           async with httpx.AsyncClient() as client:
               headers = {"Authorization": f"Bearer {ONE_INCH_API_KEY}"} # Add Authorization header
               response = await client.get(
                   f"https://api.1inch.dev/token/v1.2/{cls.ETHEREUM_CHAIN_ID}",
                   headers=headers # Send headers
               )
               response.raise_for_status()
               tokens_data = response.json()
              
               # 1inch API returns a dict where keys are contract addresses
               for addr, token_data in tokens_data.items():
                   if token_data.get("symbol", "").upper() == symbol_upper:
                       token_info = TokenInfo(
                           symbol=token_data["symbol"],
                           address=token_data["address"],
                           decimals=token_data["decimals"],
                       )
                       cls._token_cache[symbol_upper] = token_info
                       return token_info


       except httpx.HTTPStatusError as e:
           print(f"1inch Token API HTTP error for {symbol_upper}: {e.response.status_code} - {e.response.text}")
       except httpx.RequestError as e:
           print(f"1inch Token API request failed for {symbol_upper}: {e}")
       except Exception as e:
           print(f"Unexpected error with 1inch Token API for {symbol_upper}: {e}")


       # If both fail, return None
       return None


# ---
# ## Strategy Classifier
# This component uses Groq to understand and classify user intents.
# ---
class StrategyClassifier:
   def __init__(self):
       self.strategies = {
           StrategyType.LIMIT_ORDER: {
               "description": "Create limit orders to buy/sell at specific prices",
               "keywords": ["limit", "order", "price", "target", "when reaches", "at price"],
               "intents": [Intent.BUY, Intent.SELL, Intent.SWAP]
           },
           StrategyType.MARKET_MAKING: {
               "description": "Provide liquidity by placing buy and sell orders",
               "keywords": ["market making", "liquidity", "spread", "both sides", "provide"],
               "intents": [Intent.PROVIDE_LIQUIDITY]
           }
       }


   async def classify_intent(self, query: str) -> ClassifiedIntent:
       """Use Groq to classify user intent"""
          
       system_prompt = """
        You are an AI assistant that classifies DeFi trading intents. Based on the user query, extract:
        1. Strategy type (limit_order or market_making)
        2. Intent (buy, sell, provide_liquidity, swap)
        3. Trigger type (price_based, time_based, immediate, percentage_based)
        4. Action (create_order, cancel_order, update_order)
        5. Asset details (asset_from, asset_to, amount, price_target). For 'buy X if price Y' queries, if the 'asset_from' is not explicitly mentioned, assume the user is trading a common stablecoin (like USDC) to acquire 'X'.
        6. Crucially: The Ethereum address where the 'asset_to' (the token being received) should be sent. If not explicitly mentioned, assume it should be sent to the 'maker' (the wallet that placed the order), which should be represented by null. Do NOT output "0x0000000000000000000000000000000000000000" if the user doesn't specify a receiver, just use null.
            
        Available strategies:
        - limit_order: Buy/sell at specific price levels
        - market_making: Provide liquidity with buy/sell orders
            
        Respond only with valid JSON in this format:
        {
            "strategy": "limit_order|market_making",
            "intent": "buy|sell|provide_liquidity|swap",
            "trigger": "price_based|time_based|immediate|percentage_based",    
            "action": "create_order|cancel_order|update_order",
            "asset_from": "token_symbol",
            "asset_to": "token_symbol",    
            "amount": "amount_string_or_null",
            "price_target": "price_string_or_null",
            "trigger_value": "trigger_value_or_null",
            "receiver_address": "ethereum_address_string_or_null",
            "confidence_score": 0.0-1.0
        }
        
        Example queries and expected JSON:
        - "Buy 100 USDC with USDT when price reaches $0.99 and send the USDC to 0x123...abc": {"strategy": "limit_order", "intent": "buy", "trigger": "price_based", "action": "create_order", "asset_from": "USDT", "asset_to": "USDC", "amount": "100", "price_target": "0.99", "trigger_value": "0.99", "receiver_address": "0x123...abc", "confidence_score": 0.9}
        - "Sell 50 USDT for USDC at $1.01": {"strategy": "limit_order", "intent": "sell", "trigger": "price_based", "action": "create_order", "asset_from": "USDT", "asset_to": "USDC", "amount": "50", "price_target": "1.01", "trigger_value": "1.01", "receiver_address": null, "confidence_score": 0.9}
        - "I want to buy 1 ETH with USDC now": {"strategy": "limit_order", "intent": "buy", "trigger": "immediate", "action": "create_order", "asset_from": "USDC", "asset_to": "ETH", "amount": "1", "price_target": null, "trigger_value": null, "receiver_address": null, "confidence_score": 0.9}
        - "Send 0.5 USDC to 0xSomeoneElseAddress if I sell 1 WETH at $2500": {"strategy": "limit_order", "intent": "sell", "trigger": "price_based", "action": "create_order", "asset_from": "WETH", "asset_to": "USDC", "amount": "1", "price_target": "2500", "trigger_value": "2500", "receiver_address": "0xSomeoneElseAddress", "confidence_score": 0.95}
        - "buy 10 usdt if its goes below 0.98 $": {"strategy": "limit_order", "intent": "buy", "trigger": "price_based", "action": "create_order", "asset_from": "USDC", "asset_to": "USDT", "amount": "10", "price_target": "0.98", "trigger_value": "0.98", "receiver_address": null, "confidence_score": 0.9}
        - "buy 10 eth if its goes below 2000 $": {"strategy": "limit_order", "intent": "buy", "trigger": "price_based", "action": "create_order", "asset_from": "USDC", "asset_to": "ETH", "amount": "10", "price_target": "2000", "trigger_value": "2000", "receiver_address": null, "confidence_score": 0.9}
        """
          
       try:
           response = groq_client.chat.completions.create(
               model="llama3-70b-8192", # You can try other models like "llama3-8b-8192" or "llama3-70b-8192"
               messages=[
                   {"role": "system", "content": system_prompt},
                   {"role": "user", "content": query}
               ],
               temperature=0.1,
               max_tokens=500
           )
          
           result = json.loads(response.choices[0].message.content)
           return ClassifiedIntent(**result)
          
       except Exception as e:
           # Fallback classification in case of Groq API issues
           print(f"Groq classification failed: {e}. Falling back to rule-based.")
           return await self._fallback_classification(query)
      
   async def _fallback_classification(self, query: str) -> ClassifiedIntent:
       """Fallback rule-based classification (less precise than Groq)"""
       query_lower = query.lower()
      
       # Detect strategy
       if any(keyword in query_lower for keyword in ["market making", "liquidity", "spread"]):
           strategy = StrategyType.MARKET_MAKING
           intent = Intent.PROVIDE_LIQUIDITY
       else:
           strategy = StrategyType.LIMIT_ORDER
           if any(keyword in query_lower for keyword in ["buy", "purchase"]):
               intent = Intent.BUY
           elif any(keyword in query_lower for keyword in ["sell"]):
               intent = Intent.SELL
           else:
               intent = Intent.SWAP
      
       # Detect trigger
       if any(keyword in query_lower for keyword in ["when", "if", "reaches", "above", "below"]):
           trigger = TriggerType.PRICE_BASED
       elif any(keyword in query_lower for keyword in ["now", "immediately"]):
           trigger = TriggerType.IMMEDIATE
       else:
           trigger = TriggerType.PRICE_BASED
      
       token_symbols_found = re.findall(r'\b([A-Z]{2,5})\b', query.upper())
        
       asset_from = None # Start with None, let detection logic fill it
       asset_to = None

        # Try to infer based on common patterns and found symbols
       if "buy" in query_lower:
            # Example: "buy 1 ETH with USDC" -> asset_to=ETH, asset_from=USDC
            if "with" in query_lower:
                match = re.search(r"buy \d+(\.\d+)?\s*([A-Z]{2,5})\s+with\s+([A-Z]{2,5})", query, re.IGNORECASE)
                if match:
                    asset_to = match.group(2)
                    asset_from = match.group(3)
            elif len(token_symbols_found) >= 1: # At least one token mentioned
                asset_to = token_symbols_found[0] # The asset being bought
                # If 'asset_from' not explicitly stated, default to USDC for buying
                asset_from = "USDC" 
                # If there's a second token, and it's not the same as asset_to, assume it's asset_from
                if len(token_symbols_found) >= 2 and token_symbols_found[1] != asset_to:
                     asset_from = token_symbols_found[1] # Override default if another token found
            else:
                asset_to = "ETH" # Default for "buy" if no token mentioned
                asset_from = "USDC"

       elif "sell" in query_lower:
             # Example: "sell 1 USDT for USDC" -> asset_from=USDT, asset_to=USDC
             if "for" in query_lower:
                match = re.search(r"sell \d+(\.\d+)?\s*([A-Z]{2,5})\s+for\s+([A-Z]{2,5})", query, re.IGNORECASE)
                if match:
                    asset_from = match.group(2)
                    asset_to = match.group(3)
             elif len(token_symbols_found) >= 1: # At least one token mentioned
                asset_from = token_symbols_found[0] # The asset being sold
                # If 'asset_to' not explicitly stated, default to USDC for selling
                asset_to = "USDC" 
                # If there's a second token, and it's not the same as asset_from, assume it's asset_to
                if len(token_symbols_found) >= 2 and token_symbols_found[1] != asset_from:
                     asset_to = token_symbols_found[1] # Override default if another token found
             else:
                 asset_from = "ETH" # Default for "sell" if no token mentioned
                 asset_to = "USDC"
       else: # Swap or other ambiguous cases
            if len(token_symbols_found) >= 2:
                asset_from = token_symbols_found[0]
                asset_to = token_symbols_found[1]
            elif len(token_symbols_found) == 1:
                asset_from = token_symbols_found[0]
                asset_to = "USDC" # Default to common pairing
            else:
                asset_from = "ETH" # Fallback if no tokens inferred
                asset_to = "USDC"


        # Extract amount and price (keep this logic largely the same)
       amounts_and_prices = re.findall(r'\b(\d+(?:\.\d+)?)\b', query)
       amount = None
       price_target = None

       if len(amounts_and_prices) > 0:
            if "sell" in query_lower and ("if the price drops to" in query_lower or "goes below" in query_lower):
                if len(amounts_and_prices) >= 1:
                    amount = amounts_and_prices[0]
                if len(amounts_and_prices) >= 2:
                    price_target = amounts_and_prices[1]
                else:
                    price_target = amounts_and_prices[0]
                    amount = "1"
            elif "buy" in query_lower and ("with" in query_lower or "for" in query_lower or "if price reaches" in query_lower or "goes below" in query_lower):
                if len(amounts_and_prices) >= 1:
                    amount = amounts_and_prices[0]
                if len(amounts_and_prices) >= 2:
                    price_target = amounts_and_prices[1]
            else:
                if len(amounts_and_prices) >= 1:
                    amount = amounts_and_prices[0]
                if len(amounts_and_prices) >= 2:
                    price_target = amounts_and_prices[1]
      
       # Ensure asset_from and asset_to are not the same unless explicitly intended (e.g. for wrapping)
       if asset_from.upper() == asset_to.upper() and asset_from.upper() not in ["ETH", "WETH"]:
           # If they are the same and not ETH/WETH pair, it's likely a misidentification
           # Default to common pair for fallback clarity
           if intent == Intent.BUY:
               asset_from = "USDT"
               asset_to = "USDC"
           elif intent == Intent.SELL:
               asset_from = "USDT"
               asset_to = "USDC"


       # Extract receiver address
       # Extract receiver address (keep this)
       receiver_address = None
       address_match = re.search(r'(0x[a-fA-F0-9]{40})', query)
       if address_match:
            receiver_address = address_match.group(1)

       return ClassifiedIntent(
            strategy=strategy,
            intent=intent,
            trigger=trigger,
            action=ActionType.CREATE_ORDER,
            asset_from=asset_from or "USDC", # Ensure asset_from is set, default if still None
            asset_to=asset_to or "ETH",     # Ensure asset_to is set, default if still None
            amount=amount,
            price_target=price_target,
            trigger_value=price_target,
            receiver_address=receiver_address,
            confidence_score=0.5 # Lower confidence for fallback
        )


# ---
# ## 1inch Orderbook Service
# Handles interaction with the 1inch Orderbook API for creating and managing orders.
# ---
class OrderBookService:
   BASE_URL = "https://api.1inch.dev/orderbook/v4.0"
  
   @staticmethod
   async def get_create_order_params(intent: ClassifiedIntent, wallet_address: str) -> Dict[str, Any]:
       """Generate parameters for creating 1inch limit order"""
      
       # Get token info (address and decimals) dynamically
       maker_token_info = await TokenInfoService.get_token_info(intent.asset_from)
       taker_token_info = await TokenInfoService.get_token_info(intent.asset_to)
      
       if not maker_token_info or not taker_token_info:
           raise ValueError(f"Could not retrieve full token information for {intent.asset_from} or {intent.asset_to}.")
          
       maker_asset_address = maker_token_info.address
       taker_asset_address = taker_token_info.address
       maker_decimals = maker_token_info.decimals
       taker_decimals = taker_token_info.decimals


       # Determine the actual receiver address
       final_receiver_address = intent.receiver_address if intent.receiver_address else wallet_address


       def to_wei(amount_str: Optional[str], decimals: int) -> str:
           if amount_str is None or not amount_str.strip():
               return "0"
           try:
               amount_decimal = Decimal(amount_str)
               return str(int(amount_decimal * (Decimal(10)**decimals)))
           except InvalidOperation:
               raise ValueError(f"Invalid number format for amount: {amount_str}")
           except Exception as e:
               raise ValueError(f"Error converting amount {amount_str} to wei with {decimals} decimals: {e}")


       making_amount = "0"
       taking_amount = "0"


       # Logic for calculating makingAmount and takingAmount based on intent and price target
       if intent.amount and intent.price_target:
           try:
               amount_num = Decimal(intent.amount)
               price_num = Decimal(intent.price_target)


               if intent.intent == Intent.SELL:
                   making_amount = to_wei(str(amount_num), maker_decimals)
                   taking_amount = to_wei(str(amount_num * price_num), taker_decimals)
               elif intent.intent == Intent.BUY:
                   taking_amount = to_wei(str(amount_num), taker_decimals)
                   making_amount = to_wei(str(amount_num * price_num), maker_decimals)
              
           except InvalidOperation:
               raise ValueError("Could not parse amount or price target as a number.")
       elif intent.amount:
           # If only amount is specified, assume a 1:1 swap or implicit market order
           # This is a limit order, so a price_target is generally expected.
           # If no price_target is given, it means it's a "market order" or "immediate swap"
           # In such a case, the price would be fetched at the time of order fulfillment.
           # For a limit order, we need a price. If not specified, we can default to 1:1 if applicable
           # or raise an error asking for price_target for limit orders.
           # For now, default to 1:1 for making/taking amount if no price target is specified.
           print(f"Warning: Amount {intent.amount} specified, but no price target. Assuming 1:1 swap if possible.")
           making_amount = to_wei(intent.amount, maker_decimals)
           taking_amount = to_wei(intent.amount, taker_decimals)
       else: # Handle cases where amount is missing but price_target might exist
           print(f"Warning: No amount specified for intent. Defaulting to a small amount (1) for calculation.")
           default_amount_for_calc = "1" # Default amount to use for calculation if not specified
           if intent.price_target:
               try:
                   price_num = Decimal(intent.price_target)
                   if intent.intent == Intent.SELL:
                       making_amount = to_wei(default_amount_for_calc, maker_decimals)
                       taking_amount = to_wei(str(Decimal(default_amount_for_calc) * price_num), taker_decimals)
                   elif intent.intent == Intent.BUY:
                       taking_amount = to_wei(default_amount_for_calc, taker_decimals)
                       making_amount = to_wei(str(Decimal(default_amount_for_calc) * price_num), maker_decimals)
               except InvalidOperation:
                   raise ValueError("Could not parse price target as a number.")
           else:
                # If neither amount nor price target, this order is underspecified.
                raise ValueError("Order requires at least an amount or a price target to be defined.")




       chain_id = TokenInfoService.ETHEREUM_CHAIN_ID


       return {
           "endpoint": f"{OrderBookService.BASE_URL}/{chain_id}",
           "method": "POST",
           "headers": {
               "Authorization": f"Bearer {ONE_INCH_API_KEY}", # API key is required for POSTing orders
               "accept": "application/json",
               "content-type": "application/json"
           },
           "payload": {
               "orderHash": "to_be_calculated",
               "signature": "to_be_signed",
               "data": {
                   "makerAsset": maker_asset_address,
                   "takerAsset": taker_asset_address,
                   "maker": wallet_address,
                   "receiver": final_receiver_address,
                   "makingAmount": making_amount,
                   "takingAmount": taking_amount,
                   "salt": str(int(time.time())),
                   "extension": "0x",
                   "makerTraits": "0"
               }
           }
       }
  
   @staticmethod
   def get_orders_params(wallet_address: str, asset_filter: Optional[str] = None) -> Dict[str, Any]:
       """Generate parameters for fetching user orders"""
       chain_id = TokenInfoService.ETHEREUM_CHAIN_ID
       params = {
           "page": 1,
           "limit": 100,
           "statuses": "1,2,3" # Active, Filled, PartialFilled
       }
      
       if asset_filter:
           # For fetching orders, you usually filter by contract address, not symbol
           # This would require fetching token info dynamically here too
           # token_info = await TokenInfoService.get_token_info(asset_filter)
           # if token_info:
           #     params["makerAsset"] = token_info.address
           pass


       query_string = "&".join([f"{k}={v}" for k, v in params.items()])
      
       return {
           "endpoint": f"{OrderBookService.BASE_URL}/{chain_id}/address/{wallet_address}?{query_string}",
           "method": "GET",
           "headers": {
               "Authorization": f"Bearer {ONE_INCH_API_KEY}", # API key is required for fetching orders
               "accept": "application/json",
               "content-type": "application/json"
           }
       }


classifier = StrategyClassifier()


# ---
# ## Main API Endpoint
# This endpoint processes natural language queries and translates them into actionable 1inch API parameters.
# ---
@app.post("/process-intent", response_model=APIResponse)
async def process_user_intent(request: UserQuery):
   """Main endpoint to process natural language queries"""
      
   try:
       # Classify the user intent
       intent = await classifier.classify_intent(request.query)
       print(intent)
      
       # If user explicitly provided receiver_address in the request body, override Groq's extraction
       if request.receiver_address:
           intent.receiver_address = request.receiver_address


       # Validate strategy support
       if intent.strategy not in [StrategyType.LIMIT_ORDER, StrategyType.MARKET_MAKING]:
           return APIResponse(
               success=False,
               error_message="Unsupported strategy",
               guidance="Currently supported strategies: Limit Orders and Market Making. Please rephrase your request."
           )
          
       # Get token addresses and decimals dynamically before creating order parameters
       try:
           from_token_info = await TokenInfoService.get_token_info(intent.asset_from)
           to_token_info = await TokenInfoService.get_token_info(intent.asset_to)
           print(from_token_info,to_token_info)
       except Exception as e:
           return APIResponse(
               success=False,
               error_message=f"Failed to fetch token information: {e}",
               guidance="Please ensure token symbols are correct (e.g., USDT, USDC, ETH)."
           )


       if not from_token_info or not to_token_info:
           return APIResponse(
               success=False,
               error_message=f"Could not find information for tokens: {intent.asset_from} or {intent.asset_to}.",
               guidance="Please provide valid and supported token symbols like USDT, USDC, ETH, etc."
           )
          
       # Generate API parameters based on strategy
       if intent.strategy == StrategyType.LIMIT_ORDER:
           if intent.action == ActionType.CREATE_ORDER:
               api_params = await OrderBookService.get_create_order_params(intent, request.wallet_address)
               return APIResponse(
                   success=True,
                   intent=intent,
                   api_endpoint=api_params["endpoint"],
                   api_method=api_params["method"],
                   api_parameters=api_params
               )
           else:
               # For cancel/update operations (simplified, needs more logic for orderHash etc.)
               api_params = OrderBookService.get_orders_params(request.wallet_address, intent.asset_from)
               return APIResponse(
                   success=True,
                   intent=intent,
                   api_endpoint=api_params["endpoint"],
                   api_method=api_params["method"],
                   api_parameters=api_params,
                   guidance="Order cancellation/update logic needs specific order identifiers."
               )
          
       elif intent.strategy == StrategyType.MARKET_MAKING:
           return APIResponse(
               success=True,
               intent=intent,
               guidance="Market making strategy requires creating multiple limit orders. Please specify price ranges and amounts."
           )
          
   except ValueError as ve:
       return APIResponse(
           success=False,
           error_message=f"Parameter error: {str(ve)}",
           guidance="Please check your input values for amounts and addresses."
       )
   except Exception as e:
       return APIResponse(
           success=False,
           error_message=f"Processing error: {str(e)}",
           guidance="Please rephrase your request or contact support."
       )


# ---
# ## Supported Strategies Endpoint
# Provides information about the trading strategies this application supports.
# ---
@app.get("/supported-strategies")
async def get_supported_strategies():
   """Get list of supported strategies"""
   return {
       "strategies": [
           {
               "type": "limit_order",
               "description": "Create buy/sell orders at specific price levels",
               "examples": [
                   "Buy 100 USDC with USDT when price reaches $0.99",
                   "Sell 50 USDT for USDC at $1.01",
                   "Create limit order to swap 200 USDC to USDT",
                   "Buy 1 ETH with USDC and send it to 0xAbCdEf1234567890abcdef1234567890abcdef",
                   "Send 0.5 USDC to 0xSomeoneElseAddress if I sell 1 WETH at $2500"
               ]
           },
           {
               "type": "market_making",   
               "description": "Provide liquidity by placing orders on both sides",
               "examples": [
                   "Provide liquidity for USDT/USDC pair with 0.1% spread",
                   "Market make between USDT and USDC with $1000"
               ]
           }
       ]
   }




## Supported Tokens Endpoint
# Dynamically fetches and lists tokens supported by the 1inch Orderbook API on Ethereum.
# ---
@app.get("/supported-tokens")
async def get_supported_tokens():
   """Get a list of tokens supported by 1inch on Ethereum (fetched dynamically)"""
   try:
       async with httpx.AsyncClient() as client:
           headers = {"Authorization": f"Bearer {ONE_INCH_API_KEY}"} # REQUIRED for this endpoint
           response = await client.get(
               f"https://api.1inch.dev/token/v1.2/{TokenInfoService.ETHEREUM_CHAIN_ID}",
               headers=headers # Pass the Authorization header
           )
           response.raise_for_status()
           tokens_data = response.json()
          
           formatted_tokens = []
           for addr, token_info in tokens_data.items():
               formatted_tokens.append({
                   "symbol": token_info.get("symbol"),
                   "name": token_info.get("name"),
                   "address": token_info.get("address"),
                   "decimals": token_info.get("decimals"),
                   "network": "Ethereum"
               })
           return {"tokens": formatted_tokens}
   except httpx.HTTPStatusError as e:
       # Include more detail from the 1inch error response if available
       error_detail = e.response.json() if e.response.text else "No specific error message from 1inch API."
       raise HTTPException(status_code=e.response.status_code, detail=f"1inch API error: {error_detail}")
   except httpx.RequestError as e:
       raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Failed to connect to 1inch API: {e}")
   except Exception as e:
       raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An unexpected error occurred: {e}")




if __name__ == "__main__":
   import uvicorn
   # Make sure your Groq and 1inch API keys are correctly set above
   uvicorn.run(app, host="0.0.0.0", port=8004)

