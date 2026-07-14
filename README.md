# testing_activation_layers_llms
In this mini-project we tested activations layers in Qwen 7B 2.5 Instruct to check anomalies in activation norms and layers when infiltrating poisoned data.

Test cases & datasets used:
- World Cup matches 1930-2022 (~1000 rows): https://www.kaggle.com/datasets/piterfm/fifa-football-world-cup?resource=download&select=matches_1930_2022.csv 
- LaLiga Results from 1995 to 2025 (~11k rows): https://www.kaggle.com/datasets/kishan305/la-liga-results-19952020/data

LLM Model Used: Qwen 2.5 7B Instruct - https://huggingface.co/Qwen/Qwen2.5-7B-Instruct 

Distribution Model used (for calculating match results): joint probability mass function (PMF) of two independent Poisson random variables.

References: 
1.	https://www.kaggle.com/datasets/piterfm/fifa-football-world-cup?resource=download&select=matches_1930_2022.csv
2.	Maher, M.J.: Modelling association football scores. Statistica Neerlandica 36(3), 109–118 (1982)
3.	Dixon, M.J., Coles, S.G.: Modelling association football scores and inefficiencies in the football betting market. Journal of the Royal Statistical Society: Series C (Applied Statistics) 46(2), 265–280 (1997)
4.	https://www.kaggle.com/datasets/kishan305/la-liga-results-19952020/data
