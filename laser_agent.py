import argparse
import random
import string
import openai
from dotenv import load_dotenv
load_dotenv()

import adatest
import re
import os
import sys
sys.path.insert(0, 'path_to_WebShop_repo')
from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv

cache_dir = 'your_own_cache_dir'
import sys
import time
import functools
from joblib import Memory
import json
import openai.error
import backoff
from copy import deepcopy
from prompt_library import *
import numpy as np
random.seed(233)
np.random.seed(233)
CacheMemory = Memory(location=cache_dir, verbose=0)
def backoff_hdlr(details):
    # Handler from https://pypi.org/project/backoff/
    print("Backing off {wait:0.1f} seconds after {tries} tries "
          "calling function {target} with args {args} and kwargs "
          "{kwargs}".format(**details))

with open(os.path.expanduser('your_openai_key_file'), 'r') as file:
    openai.api_key = file.read().replace('\n', '')

class OpenAIModel(adatest.Model):
    def __init__(self, model="gpt-4-0613", quote="\"", temperature=0.7, top_p=1, max_length=30, n=1, logprobs=None):

        self.model_name = model
        self.model = model
        self.api_key = openai.api_key
        self.quote = quote
        self.temperature = temperature
        self.top_p = top_p
        self.max_length = max_length
        self.n = n
        self.logprobs = logprobs

    @functools.lru_cache(maxsize=None)
    @CacheMemory.cache
    @backoff.on_exception(backoff.expo,
                          (openai.error.RateLimitError,
                           openai.error.ServiceUnavailableError),
                          max_time=1000,
                          on_backoff=backoff_hdlr)
    def __call__(self, messages, functions=None, function_call=None, salient=True):
        time.sleep(1)
        messages = eval(messages)
        if functions is not None:
            functions = eval(functions)
        if function_call is not None:
            function_call = eval(function_call)
        if not salient:
            print ('messages', messages)
        if functions is not None:
            resp = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_length,
                temperature=self.temperature,
                top_p=self.top_p,
                n=self.n,
                stop=self.quote,
                functions=functions, function_call=function_call if function_call is not None else "auto",
            )
        else:
            resp = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_length,
                temperature=self.temperature,
                top_p=self.top_p,
                n=self.n,
                stop=self.quote,
            )
        if not salient:
            print (resp)
        content = [x["message"]['content'] for x in resp['choices']]
        if 'function_call' in resp['choices'][0]['message']:
            action = [x["message"]['function_call'] for x in resp['choices']]
        else:
            action = None

        return content, action

def get_env():
    env = WebAgentTextEnv(
            observation_mode='text_rich',
            render=False,
            num_products=None, # 1000 for small product space, None for full product space
            human_goals=True,
        )
    return env

def parse_customization(response):
    if ':' in response:
        options = response.split(':')[1].strip().split(',')
    else:
        options = response.split(',')
    options = [o.strip() for o in options]
    options = [o for o in options if o.lower() != 'none']
    return options

def auxilary_get_action(model, ob, response, available_options, setup, force=None):
    for i in range(3):
        aux_prompt = deepcopy(chat_zero_shot_mapping_action_prompt_gpt4)
        aux_prompt[0]['content'] = aux_prompt[0]['content'] % (setup[0], setup[1], setup[3])
        aux_prompt[-1]['content'] = aux_prompt[-1]['content'] % ob + '\n\n' + 'Next action rationale: ' + response[0]
        #print ('Before auxiliary prompt', response)
        if not force:
            response, action = model(str(aux_prompt), functions=str(available_options))
        else:
            response, action = model(str(aux_prompt), functions=str(available_options), function_call=force)
        if action is not None:
            return response, action
        else:
            try:
                if type(eval(response[0])) == dict:
                    #print (response)
                    res = eval(response[0])
                    #print ('The action is returned in response')
                    keys = list(res.keys())
                    action = [{'name': res[keys[0]].replace('functions.', ''), 'arguments': str(res[keys[1]])}]
                    return response, action
            except:
                pass
    return None, None

def item_page_agent(env, ob, model, item_config):
    available_options = [description, reviews, features, buy_item, previous_page]
    additional_info = []
    mapping = {'Description': description, 'Reviews': reviews, 'Features': features}
    name2str = {'Description': 'description', 'Reviews': 'reviews', 'Features': 'features'}
    thinking = []
    item_text = [ob.strip().split('\n')[-7].strip(), ob.strip().split('\n')[-6].strip()]
    print (ob)
    while len(available_options) > 1:
        prompt = deepcopy(chat_zero_shot_indiv_prompt_gpt4)
        item_details = ["Target item details:"] + [f"{k}: {v}" for k, v in item_config.items()]
        item_details = '\n'.join(item_details)
        if len(additional_info) > 0:
            ob = '\n'.join([ob] + additional_info)
        ob = 'Current observation:\n' + ob + '\n' + item_details
        prompt[0]['content'] = prompt[0]['content'] % tuple(web_shop_verify_gpt4)
        prompt[-1]['content'] = prompt[-1]['content'] % ob
        response, action = model(str(prompt))
        response = post_process_response(response)
        if response[0] is not None:
            thinking.extend(response)
        if action is None:
            _, action = auxilary_get_action(model, ob, response, available_options, web_shop_verify_gpt4)
            if action == None:
                # This probably only happens when the model wanted to do customization, which we do not provide at this step
                action = [{'name': 'Buy_Now'}]
        if action[0]['name'] == 'Buy_Now':
            custom_prompt = deepcopy(chat_zero_shot_custom_prompt)
            custom_prompt[-1]['content'] = custom_prompt[-1]['content'] % ob
            tmp_response, _ = model(str(custom_prompt))
            options = parse_customization(tmp_response[0])
            arg_list = {}
            for op in options:
                arg_list[op] = {'type': 'string', 'description': f"The {op} of the item"}
            buy_item_final['parameters']['properties'] = arg_list
            buy_item_final['parameters']['required'] = list(arg_list.keys())
            prompt = deepcopy(chat_zero_shot_indiv_prompt_gpt4)
            web_shop_buy = deepcopy(web_shop_verify_gpt4)
            web_shop_buy[-1] = """At this stage, you have found the correct item. You task is to generate the correct customization options of the current item to best match the user instruction. Prepare your response in the following format:
Rationale: the user wanted {keywords of the target item}, and they required the following customization options: {cutomization of the target item}, the current item has the following customization options: {options available for the current item}, thus we should choose {the correct customization options}"""
            prompt[0]['content'] = prompt[0]['content'] % tuple(web_shop_buy)
            prompt[-1]['content'] = prompt[-1]['content'] % ob
            final_response, action = model(str(prompt))
            if action is None:
                _, action = auxilary_get_action(model, ob, final_response, str([buy_item_final]), web_shop_buy, force=str({'name': 'Buy_Now'}))
            if action is not None:
                selections = eval(action[0]['arguments'])
                for k, v in selections.items():
                    if v:
                        act = f"click[{v}]"
                        ob, rew, done, _ = env.step(act)
            act = "click[Buy Now]"            
            return act, thinking, item_text+thinking[-1:]
        elif action[0]['name'] == 'Prev':
            act = "click[< Prev]"
            print ('********* page agent decide to go back')
            return act, thinking, item_text+thinking[-1:]
        elif action[0]['name'] == 'Back_to_Search':
            act = "click[Back to Search]"
            return act, thinking, item_text+thinking[-1:]
        else:
            if action[0]['name'] not in mapping:
                continue
            act_name = mapping[action[0]['name']]
            if act_name in available_options:
                available_options.remove(act_name)
            else:
                #print ('the option is not available', action[0]['name'])
                act = "click[< Prev]"
                print ('********* page agent forced to go back')
                return act, thinking, item_text+thinking[-1:]
            act = f"click[{name2str[action[0]['name']]}]"
        ob, rew, done, _ = env.step(act)
        template = r"\[button\] < Prev \[button_\]\n(.+)"
        match = re.search(template, ob)
        if match is not None:
            new_info = re.search(template, ob).group(1).strip()
        else:
            new_info = 'None'
        additional_info.append(f"{name2str[action[0]['name']]}:\n{new_info}\n")
        ob, rew, done, _ = env.step("click[< Prev]")

def post_process_response(response):
    response = response[0]
    response = response.replace('Rationale:', '').strip()
    response = response.replace('Feedback:', '').strip()
    pattern = r"Rationale\d: "
    response = re.sub(pattern, '', response).strip()
    return [response]

def print_prompt(prompt):
    for turn in prompt:
        print (turn['content'])

def get_action(action, mapping):
    action[0]['arguments'] = action[0]['arguments'].replace('null', 'None').replace('false', 'False').replace('true', 'True')
    act_name = mapping[action[0]['name']]
    act_arg = eval(action[0]['arguments'])
    return act_name, act_arg

def back_up_agent(env, session, model, browsed_items, instruction):
    fake_ob = f'Current observation:\nInstruction:\n{instruction}\n'
    for k, v in browsed_items.items():
        fake_ob += f'[button] {k} [button_]\n'
        fake_ob += f'{v[0]}\n{v[1].replace("Price: ", "")}\n'
    print ('-----------------Back up agent-----------------')
    prompt = deepcopy(chat_zero_shot_indiv_prompt_gpt4)
    web_shop_backup = deepcopy(web_shop_select_gpt4)
    fake_layout = web_shop_backup[3].split('\n')
    fake_layout = '\n'.join(fake_layout[:2] + fake_layout[5:])
    web_shop_backup[3] = fake_layout
    web_shop_backup[4] = 'At this stage, you should identify one of the items on the current page that best matches the user instruction. If none of the items match the user instruction, identify the item that is the closest match to the user instruction.'
    prompt[0]['content'] = prompt[0]['content'] % tuple(web_shop_backup)
    prompt[-1]['content'] = prompt[-1]['content'] % fake_ob 
    response, action = model(str(prompt))
    response = post_process_response(response)
    if action is None:
        response, action = auxilary_get_action(model, fake_ob, response, str([click_item]), web_shop_select_gpt4, force=str({'name': 'select_item'}))
    best_item = eval(action[0]['arguments'])['item_id']
    if best_item == '':
        best_item = list(browsed_items.keys())[0]
        print ('Back up agent failed to select any item')
    (ob, _) = env.reset(session=session)
    ob, rew, done, _ = env.step(f'search[{browsed_items[best_item][0]}]')
    ob, rew, done, _ = env.step(f'click[{best_item}]')

    custom_prompt = deepcopy(chat_zero_shot_custom_prompt)
    custom_prompt[-1]['content'] = custom_prompt[-1]['content'] % ob
    tmp_response, _ = model(str(custom_prompt))
    options = parse_customization(tmp_response[0])
    arg_list = {}
    for op in options:
        arg_list[op] = {'type': 'string', 'description': f"The {op} of the item"}
    buy_item_final['parameters']['properties'] = arg_list
    buy_item_final['parameters']['required'] = list(arg_list.keys())
    prompt = deepcopy(chat_zero_shot_indiv_prompt_gpt4)
    web_shop_buy = deepcopy(web_shop_verify_gpt4)
    web_shop_buy[-1] = """At this stage, you have found the correct item. You task is to generate the correct customization options of the current item to best match the user instruction. Prepare your response in the following format:
Rationale: the user wanted {keywords of the target item}, and they required the following customization options: {cutomization of the target item}, the current item has the following customization options: {options available for the current item}, thus we should choose {the correct customization options}"""
    prompt[0]['content'] = prompt[0]['content'] % tuple(web_shop_buy)
    prompt[-1]['content'] = prompt[-1]['content'] % ob
    response, action = model(str(prompt))
    response = post_process_response(response)
    if action is None:
        response, action = auxilary_get_action(model, ob, response, str([buy_item_final]), web_shop_buy, force=str({'name': 'Buy_Now'}))
    action[0]['arguments'] = action[0]['arguments'].replace('null', 'None').replace('false', 'False').replace('true', 'True')
    selections = eval(action[0]['arguments'])
    for k, v in selections.items():
        if v:
            act = f"click[{v}]"
            ob, rew, done, _ = env.step(act)
    act = "click[Buy Now]"
    ob, rew, done, _ = env.step(act)
    return rew  

def indiv_prompt_agent(agent_args):
    env = get_env()
    max_iter = 13
    rewards = []
    term_status = []
    name = 'gpt-4-0613'
    model = OpenAIModel(model=name, quote='\n\n', temperature=0, max_length=200, n=1)
    start = agent_args.start
    end = start + agent_args.num_examples
    episodes_len = {}
    for session in range(start, end):
        (ob, _) = env.reset(session=session)
        available_funcs = None
        forced_funcs = None
        mapping = {'Search': 'search', 'select_item': 'click', 'Next': 'next', 'Prev': 'prev', 'Back_to_Search': 'back',
                'search_item_with_history': 'search'}
        history = ['History:']
        thinking = []
        item_config = None
        browsed_items = {}
        bought = False
        instruction = None 
        counter = 0
        while counter < max_iter:
            ob = '\n'.join(ob.strip().split('\n\n'))
            ob = 'Current observation:\n' + ob
            prompt = deepcopy(chat_zero_shot_indiv_prompt_gpt4) 
            if 'Back to Search' in ob:
                if counter == 1:
                    print (ob)
                available_funcs = [click_item, next_page, back_to_search]
                setup = web_shop_select_gpt4
            else:
                if not instruction:
                    instruction = ob.split('\n')[3].strip()
                    print (instruction)
                if len(history) == 1:
                    available_funcs = [search_items]
                    setup = web_shop_search_gpt4
                else:
                    available_funcs = [search_items]
                    setup = web_shop_search_gpt4
                    ob = ob + '\n' + '\n'.join(history)
                
            prompt[0]['content'] = prompt[0]['content'] % tuple(setup)
            prompt[-1]['content'] = prompt[-1]['content'] % ob 
            if 'Back to Search' not in ob and len(history) > 1:
                print_prompt(prompt)
                history = ['History:']
            response, action = model(str(prompt))
            response = post_process_response(response)
            if response[0] is not None:
                thinking.extend(response)
            else:
                thinking.extend(['None'])
            if action is None:
                _, action = auxilary_get_action(model, ob, response, available_funcs, setup)
            act_name, act_arg = get_action(action, mapping)
            if act_name == 'search':
                args = ', '.join([f"{k}='{v}'" for k, v in act_arg.items()])
                history.append(f'Rationale{int((len(history)-1)/2)}: {thinking[-1]}')
                history.append(f'Action{int((len(history)-1)/2)}: {action[0]["name"]}({args})')
                item_config = act_arg
                act = f"{act_name}[{act_arg['keywords']}]"
            elif act_name == 'click' and act_arg['item_id'] != 'next':
                print ('going to check', act_arg['item_id'])
                counter += 1
                if act_arg['item_id'] in browsed_items:
                    print ('Agent attempt to click an item that has been clicked before')
                    aux_prompt = deepcopy(chat_zero_shot_mapping_action_prompt_gpt4)
                    aux_prompt[0]['content'] = aux_prompt[0]['content'] % (setup[0], setup[1], setup[3])
                    aux_prompt[-1]['content'] = aux_prompt[-1]['content'] % ob + '\n\n' + 'Next action rationale: ' + f'items that have been clicked before do not match the user instruction, so the next action should be select a different item that have not been clicked before.'
                    response, action = model(str(aux_prompt), functions=str(available_funcs), function_call=forced_funcs)
                    act_name = mapping[action[0]['name']]
                    act_arg = eval(action[0]['arguments'])
                if act_name == 'next':
                    act = "click[Next >]"
                    history.append(f'Rationale{int((len(history)-1)/2)}: {thinking[-1]}')
                    history.append(f'Action{int((len(history)-1)/2)}: next_page()')
                else:
                    history.append(f'Rationale{int((len(history)-1)/2)}: {thinking[-1]}')
                    history.append(f'Action{int((len(history)-1)/2)}: {action[0]["name"]}({act_arg["item_id"]})')
                    act = f"{act_name}[{act_arg['item_id']}]"
                    ob, rew, done, _ = env.step(act)
                    act, rationale, item_text = item_page_agent(env, ob, model, item_config)
                    thinking.extend(rationale)
                    browsed_items[act_arg['item_id']] = item_text
                    if act == "click[< Prev]":
                        history.append(f'Rationale{int((len(history)-1)/2)}: {thinking[-1]}')
                        history.append(f'Action{int((len(history)-1)/2)}: previous_page()')
            elif act_name == 'next':
                act = "click[Next >]"
                history.append(f'Rationale{int((len(history)-1)/2)}: {thinking[-1]}')
                history.append(f'Action{int((len(history)-1)/2)}: next_page()')
            elif act_name == 'prev':
                act = "click[< Prev]"
                history.append(f'Rationale{int((len(history)-1)/2)}: {thinking[-1]}')
                history.append(f'Action{int((len(history)-1)/2)}: previous_page()')
            elif act_name == 'back':
                act = "click[Back to Search]"
                history.append(f'Rationale{int((len(history)-1)/2)}: {thinking[-1]}')
                history.append(f'Action{int((len(history)-1)/2)}: back_to_search()')
            else:
                print ('encountered unknown action', act_name)
                exit()
            counter += 1
            ob, rew, done, _ = env.step(act)
            if act == 'click[Buy Now]':
                bought = True
                break
            #print ('iter', counter)
        if not bought:
            rew = back_up_agent(env, session, model, browsed_items, instruction)
            episodes_len[session] = -1 
        else:
            episodes_len[session] = counter
        rewards.append(rew)
        term_status.append(rew==1)
        print (f'episode {session}, reward {rew}, term_status {rew==1}')
    print('reward', np.mean(rewards), 'term_status', np.mean(term_status))
    with open(f'{name}_rewards_{start}-{end}_max13.json', 'w') as fout:
        json.dump(rewards, fout)
    with open(f'{name}_episodes_len_{start}-{end}_max13.json', 'w') as fout:
        json.dump(episodes_len, fout)
    return rewards

if __name__ == '__main__':
    parser  = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, default="webshop")
    parser.add_argument("--model_name", type=str, choices=["gpt4-0613"], default="gpt4-0613")
    parser.add_argument("--num_examples", type=int, default=10)
    parser.add_argument("--start", type=int, default=0)
    args = parser.parse_args()
    indiv_prompt_agent(args)