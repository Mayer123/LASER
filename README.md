# LASER: LLM Agent with State-Space Exploration for Web Navigation
The code for the paper "LASER: LLM Agent with State-Space Exploration for Web Navigation". See full paper [here](https://arxiv.org/abs/2309.08172)

## Enviroments
Please follow WebShop repo to set up the environment [here](https://github.com/princeton-nlp/WebShop)

After setting up the WebShop environment, please replace the web_agent_site folder with the one in this repo. We simply added seeds everywhere to make sure the instructions are deterministic to save OpenAI API calls (every call is cached). 

## Running LASER 
You can control how many episodes to evaluate by changing the num_examples flag. The default is 10.
```
python laser_agent.py --num_examples 500
```

## Cite 
```
@misc{ma2023laser,
    title={LASER: LLM Agent with State-Space Exploration for Web Navigation},
    author={Kaixin Ma and Hongming Zhang and Hongwei Wang and Xiaoman Pan and Dong Yu},
    year={2023},
    eprint={2309.08172},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```