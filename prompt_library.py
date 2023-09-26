search_items = {
      "name": "Search",
      "description": "Use this function to search for the target item in the inventory based on keywords",
      "parameters": {
        "type": "object",
        "properties": {
          "keywords": {
            "type": "string",
            "description": "The keywords that describe the item to be searched for"
          },
          "max_price": {
            "type": "string",
            "description": "The upper bound of the item price, if the upper bound is not specified, then set to 1000000.",
          }
        },
        "required": ["keywords"]
      }
    }

click_item = {
      "name": "select_item",
      "description": "Use this function to select one of the items from the search results and check its details",
      "parameters": {
        "type": "object",
        "properties": {
          "item_id": {
            "type": "string",
            "description": "The id of the item to be checked"
          },
        },
        "required": ["item_id"]
      }
    }

description = {
      "name": "Description",
      "description": "Use this function to check the description of the item, if you are unsure if the item perfectly matches the user instruction",
      "parameters": {
        "type": "object",
        "properties": {
        },
        "required": []
      }
}

features = {
      "name": "Features",
      "description": "Use this fucntion to check the features of the item, if you are unsure if the item perfectly matches the user instruction",
      "parameters": {
        "type": "object",
        "properties": {
        },
        "required": []
      }
}

reviews = {
      "name": "Reviews",
      "description": "Use this function to check the reviews of the item, if you are unsure if the item perfectly matches the user instruction",
      "parameters": {
        "type": "object",
        "properties": {
        },
        "required": []
      }
}

buy_item = {
      "name": "Buy_Now",
      "description": "Use this function to buy the current item, if the current item perfectly matches the user instruction.",
      "parameters": {
        "type": "object",
        "properties": {
        },
        "required": []
      }
    }

buy_item_final = {
      "name": "Buy_Now",
      "description": "Use this function to buy the current item, if the current item perfectly matches the user instruction. Additionally, you should also provide the customizations of the item (if any)",
      "parameters": {
        "type": "object",
        "properties": {
        },
        "required": []
      }
    }

previous_page = {
      "name": "Prev",
      "description": "Use this fucntion to go back to the results page, if the current item does not match the user instruction.",
      "parameters": {
        "type": "object",
        "properties": {
        },
        "required": []
      }
}

next_page = {
      "name": "Next",
      "description": "Use this function to go to the next page of search results to view more items, if none of the items on the current page match the user instruction.",
      "parameters": {
        "type": "object",
        "properties": {
        },
        "required": []
      }
}

back_to_search = {
      "name": "Back_to_Search",
      "description": "Use this function to go back to the initial search page. You should use this function only if you have browsed mutliple pages of items and checked multiple items' details in the history, and none of the items match the user instruction.",
      "parameters": {
        "type": "object",
        "properties": {
        },
        "required": []
      }
}

chat_zero_shot_custom_prompt = [
    {'role': 'system', 
     'content': 
"""You are an intelligent shopping assistant that can help users find the right item. You are given an observation of the current item, in the following format: 

Current observation:
Instruction:
{User Instruction}
[button] Back to Search [button_]
[button] < Prev [button_]
{Customization type1}:
  [button] {choice1} [button_]
  [button] {choice2} [button_]
{Customization type2}:
  [button] {choice1} [button_]
  [button] {choice2} [button_]
{Item name and details}
[button] Description [button_]
[button] Features [button_]
[button] Reviews [button_]
[button] Buy Now [button_]

Your task is to extract the customization types available for this item. You should extract the type names instead of the specific choices, and separate each type name with a comma. If there is no customization availble, say 'None'. Say nothing extra.
"""},
    {'role': 'user', 'content': """%s"""}
]


chat_zero_shot_indiv_prompt_gpt4 = [
    {'role': 'system', 
     'content': 
"""You are an intelligent %s assistant that can help users %s. You are given an observation of the current %s, in the following format: 

Current observation:
%s 

Every button in the observation represents a possible action you can take. Based on the current observation, your task is to generate a rationale about the next action you should take. %s
"""},
    {'role': 'user', 'content': """%s"""}
]

web_shop_search_gpt4 = [
    'shopping', 'find the right item', 'web navigation session',
"""WebShop
Instruction: 
{the user instruction}
[button] Search [button_] (generate a search query based on the user instruction and select this button to find relevant items)""",
'Note that if an history of past rationales and actions is provided, you should also consider the history when generating the rationale.'    
]

web_shop_select_gpt4 = [
    'shopping', 'find the right item', 'web navigation session',
"""Instruction: 
{the user instruction}
[button] Back to Search [button_] (select this button to go back to the search page)
Page {current page number} (Total results: {total number of results})
[button] Next > [button_] (select this button to go to the next page of results)
[button] {item_id 1} [button_] (select this button to view item 1's details)
{name of item 1}
{price of item 1}
[button] {item_id 2} [button_] (select this button to view item 2's details)
{name of item 2}
{price of item 2}
[button] {item_id 3} [button_] (select this button to view item 3's details)
{name of item 3}
{price of item 3}
{More items...}""",
"""At this stage, you want to select an item that might match the user instruction. Note that even if an item has non-matching details with the user instruction, it might offer different customization options to allow you to match. E.g. an item may have color x in its name, but you can customize it to color y later, the customization options are shown after you select the item. Thus if an item name seems relevant or partially matches the instruction, you should select that item to check its details. If an item has been selected before (the button has been clicked), you should not select the same item again. In other words, do not select an item with [clicked button] item_id [clicked button_]. Prepare your response in the following format:
Rationale: the user wanted {keywords of the target item}, and we have found {matching keywords of item x}, thus item {item_id x} seems to be a match."""   
]

web_shop_verify_gpt4 = [
    'shopping', 'find the right item', 'web navigation session',
"""Instruction:
{User Instruction}
[button] Back to Search [button_] (select this button to go back to the search page)
[button] < Prev [button_] (select this button to go back to the previous page of results)
{Customization type1}:
  [button] {option1} [button_] 
  [button] {option2} [button_]
{Customization type2}:
  [button] {option1} [button_]
  [button] {option2} [button_]
{more customization options... (if any)}
{Item name and details}
[button] Description [button_] (select this button to view the full description of the item)
[button] Features [button_] (select this button to view the full features of the item)
[button] Reviews [button_] (select this button to view the full reviews of the item)
[button] Buy Now [button_] (select this button to buy the item)

description: (if this is shown, the description button should not be selected again)
{full description of the item (if any) or "None"}

features: (if this is shown, the features button should not be selected again)
{full features of the item (if any) or "None"}

reviews: (if this is shown, the reviews button should not be selected again)
{full reviews of the item (if any) or "None"}

Target item details (what the user is looking for):
keywords: {keywords of the target item}
max_price: {the price of the item should not exceed this}""",
"""At this stage, you want to verify if the item matches the user instruction. You should consider the available customization options when deciding whether an item matches the user instruction. If an item can be customized to match the user instruction, or if the customization options cover the user specification, it is also a good match. If the item does not match the user instruction and it does not provide enough customization options, you can go to previous page to view other items. You can also check the item's description, features and reviews to view more details (Note that description, features and reviews could be "None", do not check them again if they are already given). Prepare your response in the following format:
Rationale: the user wanted {keywords of the target item}, and they required the following customization options: {cutomization of the target item}, the item is {keywords of the item in the current observation}, and it has the following customization options: {options available for the current item}, which {cover}/{not cover the user requirement}, thus we should {buy the item}/{check more details}/{go to previous page to view other items}"""
]

chat_zero_shot_feedback_prompt_gpt4 = [
    {'role': 'system', 
     'content': 
"""You are an intelligent %s manager that can give feedback on the action of an %s assistant. You are given an observation of the current %s and the assistant's rationale and action based on the observation, in the following format: 

Current observation:
%s

Assistant rationale: {the assistant's rationale for its next action}
Assistant action: {the assistant's next action}

Your task is to give feedback on the assistant's rationale and action. More specifically, you need to consider the following questions: does the given item perfectly matches the user's instruction? does the given item perfectly matches the target item? If the assistant is making a mistake, e.g. saying the item matches the target item but some details actually do not match, you should give detailed feedback on what is wrong. If the assistant is doing well, you should give positive feedback. 
"""},
    {'role': 'user', 'content': """%s
Assistant rationale: %s
Assistant action: %s"""}
]

chat_zero_shot_rethink_prompt_gpt4 = [
    {'role': 'system', 
     'content': 
"""You are an intelligent shopping assistant that can help users find the right item. You are given an observation of the current web navigation session, the proposed next action and its rationale, and some feedback from your manager, in the following format: 

Current observation:
%s

Assistant rationale: {the assistant's rationale for its next action}
Assistant action: {the assistant's next action}

Feedback:
{shopping manager's feedback}

Your task is to perform one of the function calls based on the feedback: %s
"""},
    {'role': 'user', 'content': """%s"""}
]

chat_zero_shot_mapping_action_prompt_gpt4 = [
    {'role': 'system', 
     'content': 
"""You are a intelligent %s assistant that can help users %s. You are given an observation of the current environment and a rationale for the next action to be taken, in the following format:

Current observation:
%s

Next action rationale: {the rationale for the next action}

Your task is to perform one of the function calls based on the rationale.
"""},
    {'role': 'user', 'content': """%s"""}
]

chat_zero_shot_manager_prompt_gpt4 = [
    {'role': 'system', 
     'content': 
"""You are an intelligent %s manager that can give feedback on the action of an %s assistant. You are given a history of the assistant's rationales and actions, an observation of the current %s and the assistant's rationale and proposed next action based on the observation, in the following format: 

History:
{a list of rationale and action pairs taken so far}

Current observation:
%s

Assistant rationale: {the assistant's rationale for its next action}
Assistant action: {the assistant's next action}

Your task is to generate feedback on the assistant's rationale and action and then suggest the next action to take. More specifically, you need to consider the following questions: does any item on the current page match user instruction? should the assistant go to the next page or go back to the search page? If the assistant is making a mistake, e.g. going to next page when the current page contains relevant items or selecting an item that's not relevant, you should give detailed feedback on what is wrong. Note that the assistant can not go to the next page if there are no items listed on the current page. If there are no items listed on the current page, you should tell the assistant to go back to search page. If the assistant is doing well, you should give positive feedback. After giving the feedback, provide a suggestion on what to do for the next step, e.g. go to the next page of results or go back to the search page.
"""},
    {'role': 'user', 'content': """%s"""}
]