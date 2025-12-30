import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns

"""Data Processing Section Section"""

df = pd.read_excel("/content/drive/MyDrive/retail_extracted/online_retail_II.xlsx", parse_dates=["InvoiceDate"])

df.info()
df.head()
df.describe(include='all')

pd.set_option("display.max_columns", 50)
pd.set_option("display.max_rows", 50)

numeric_cols = ['Quantity', 'Price']
categorical_cols = ['Invoice', 'StockCode', 'Description', 'Country']
num_returns = df[df["Quantity"] < 0].shape[0]

"""Data Cleaning and Preprocessing"""

df = df[df["Quantity"] > 0]
df = df[df["Price"] > 0]

non_product_codes = [
    "POST", "POSTAGE", "M", "C2", "CRUK", "BANK CHARGES",
    "ADJUST", "DOT", "AMAZONFEE", "S"
]

df = df[~df["StockCode"].isin(non_product_codes)]
df["StockCode"] = df["StockCode"].astype(str)
df = df[df["StockCode"].str.isdigit()]
df = df[df["Country"] == "United Kingdom"]

Q_THRESHOLD = 500
df = df[df["Quantity"] <= Q_THRESHOLD]

price_cap = df["Price"].quantile(0.95)
df = df[df["Price"] <= price_cap]

df["date"] = df["InvoiceDate"].dt.date

daily = df.groupby(["StockCode", "date"]).agg(
    daily_qty=("Quantity", "sum"),
    avg_price=("Price", "mean"),
    transactions=("Invoice", "count")
).reset_index()

daily["rolling_qty_7"] = (
    daily.groupby("StockCode")["daily_qty"].transform(lambda x: x.rolling(7, min_periods=1).mean()))

price_var = daily.groupby("StockCode")["avg_price"].nunique()
valid_price_variation = price_var[price_var >= 3].index

history_days = daily.groupby("StockCode")["date"].nunique()
valid_history = history_days[history_days >= 60].index

candidate_skus = list(set(valid_price_variation).intersection(valid_history))

"""RL Modeling Section"""

TRUE_ELASTICITY = 1.3       
EPSILON_NOISE = 0.15 
NUM_PRICE_POINTS = 40
GAMMA = 0.95
MAX_ITER = 200
BELLMAN_TOL = 1e-4

PRIOR_MU = 1.0              
PRIOR_SIGMA = 0.5          

class DynamicPricingEnv:
    def __init__(self, sku_df):
        self.sku_df = sku_df.reset_index(drop=True)
        self.days = sku_df['date'].values
        self.current_idx = 0
        
        self.min_price = float(sku_df['avg_price'].min())
        self.max_price = float(sku_df['avg_price'].max())
        self.price_grid = np.linspace(self.min_price, self.max_price, NUM_PRICE_POINTS)
        self.A = float(sku_df['rolling_qty_7'].mean())

        self.mu = PRIOR_MU
        self.sigma = PRIOR_SIGMA

        self.last_price = float(sku_df['avg_price'].iloc[0])
        self.last_qty = float(sku_df['rolling_qty_7'].iloc[0])

    def reset(self):
        self.current_idx = 0
        self.mu = PRIOR_MU
        self.sigma = PRIOR_SIGMA
        self.last_price = float(self.sku_df['avg_price'].iloc[0])
        self.last_qty = float(self.sku_df['rolling_qty_7'].iloc[0])
        return self._get_state()

    def _get_state(self):
        dow = pd.to_datetime(self.days[self.current_idx]).weekday()
        return np.array([np.log(self.last_price),
                         np.log(self.last_qty+1),
                         dow,
                         self.mu])

    def simulate_demand(self, price):
        log_q = np.log(self.A) - TRUE_ELASTICITY*np.log(price) + np.random.normal(0, EPSILON_NOISE)
        return max(np.exp(log_q), 0)

    def update_belief(self, price, qty):
        log_q = np.log(max(qty, 1e-6))
        log_p = np.log(price)
        log_A = np.log(self.A)

        y = (log_A - log_q) / log_p
        noise_var = EPSILON_NOISE**2 / (log_p**2)

        sigma_post_sq = 1 / (1/self.sigma**2 + 1/noise_var)
        mu_post = sigma_post_sq*(self.mu/self.sigma**2 + y/noise_var)

        self.mu = float(mu_post)
        self.sigma = float(np.sqrt(sigma_post_sq))

    def step(self, price):
        qty = self.simulate_demand(price)
        reward = price * qty

        self.update_belief(price, qty)
        self.last_price = price
        self.last_qty = qty
        self.current_idx += 1

        done = self.current_idx >= len(self.days)
        next_state = None if done else self._get_state()

        return next_state, reward, done


def expected_reward(env, price):
    log_q = np.log(env.A) - TRUE_ELASTICITY*np.log(price)
    expected_qty = np.exp(log_q + 0.5*EPSILON_NOISE**2)
    return price * expected_qty



# ORIGINAL VALUE ITERATION

def value_iteration(env, log_training=False):
    prices = env.price_grid
    nA = len(prices)

    V = np.zeros(nA)
    policy = np.zeros(nA)

    iter_vals, iter_deltas = [], []

    for iteration in range(MAX_ITER):
        delta = 0
        V_new = np.zeros_like(V)

        for i, p in enumerate(prices):
            rewards = np.zeros(nA)
            for j, p_next in enumerate(prices):
                rewards[j] = expected_reward(env, p_next) + GAMMA * V[j]

            best_idx = rewards.argmax()
            V_new[i] = rewards[best_idx]
            policy[i] = prices[best_idx]
            delta = max(delta, abs(V_new[i] - V[i]))

        V = V_new

        if log_training:
            iter_vals.append(np.mean(V))
            iter_deltas.append(delta)

        if delta < BELLMAN_TOL:
            break

    if log_training:
        return V, policy, iter_vals, iter_deltas

    return V, policy


#    TRAIN RL MODELS

sku_policies = {}
sku_values = {}

for sku in tqdm(candidate_skus):
    sku_df = daily[daily['StockCode']==sku].copy()
    env = DynamicPricingEnv(sku_df)
    V, policy = value_iteration(env)
    sku_policies[sku] = policy
    sku_values[sku] = V



#   TRAINING PERFORMANCE PLOT FOR ONE SKU

example_sku = candidate_skus[0]
env = DynamicPricingEnv(daily[daily['StockCode']==example_sku])
_, _, mean_vals, deltas = value_iteration(env, log_training=True)

plt.figure(figsize=(10,5))
plt.plot(mean_vals)
plt.title(f"Value Function Convergence – SKU {example_sku}")
plt.xlabel("Iteration")
plt.ylabel("Mean V")
plt.grid()
plt.show()

plt.figure(figsize=(10,5))
plt.plot(deltas)
plt.title(f"Bellman Error – SKU {example_sku}")
plt.xlabel("Iteration")
plt.ylabel("Bellman Error (log scale)")
plt.yscale("log")
plt.grid()
plt.show()



#  SIMULATE RL REVENUE

sku_revenue = {}

for sku in tqdm(candidate_skus):
    sku_df = daily[daily['StockCode']==sku].copy()
    env = DynamicPricingEnv(sku_df)
    env.reset()

    total_rev = 0
    done = False

    while not done:
        idx = np.argmin(np.abs(env.price_grid - env.last_price))
        action_price = sku_policies[sku][idx]
        _, reward, done = env.step(action_price)
        total_rev += reward

    sku_revenue[sku] = total_rev


#  ORIGINAL RL vs HISTORICAL REVENUE TABLE 

sku_list = ['21212','84077','84991','84270','84879','21977','21232','21213','84755','22197']

historical_revenue = {
    sku: (daily[daily['StockCode']==sku]['daily_qty'] *
          daily[daily['StockCode']==sku]['avg_price']).sum()
    for sku in sku_list
}

revenue_comparison = pd.DataFrame({
    'StockCode': sku_list,
    'HistoricalRevenue': [historical_revenue[sku] for sku in sku_list],
    'RL_SimulatedRevenue': [sku_revenue[sku] for sku in sku_list]
})

revenue_comparison['LiftPercent'] = (
    (revenue_comparison['RL_SimulatedRevenue'] - revenue_comparison['HistoricalRevenue'])
    / revenue_comparison['HistoricalRevenue'] * 100
)

print(revenue_comparison)

#  BASELINE MODELS FOR COMPARISON

def simulate_policy(env, price_fn):
    env.reset()
    done = False
    total = 0
    while not done:
        price = price_fn(env)
        _, reward, done = env.step(price)
        total += reward
    return total

def fixed_price_policy(env):
    return env.sku_df["avg_price"].median()

def elasticity_baseline(env):
    best_rev, best_p = -1, env.price_grid[0]
    for p in env.price_grid:
        r = expected_reward(env, p)
        if r > best_rev:
            best_rev, best_p = r, p
    return best_p

def random_baseline(env):
    return np.random.choice(env.price_grid)


baseline_rows = []

for sku in sku_list:
    sku_df = daily[daily['StockCode']==sku]
    env = DynamicPricingEnv(sku_df)

    fixed_rev = simulate_policy(env, fixed_price_policy)
    env = DynamicPricingEnv(sku_df)
    elastic_rev = simulate_policy(env, elasticity_baseline)
    env = DynamicPricingEnv(sku_df)
    random_rev = np.mean([simulate_policy(env, random_baseline) for _ in range(10)])

    baseline_rows.append({
        "StockCode": sku,
        "HistoricalRevenue": historical_revenue[sku],
        "RL Revenue": sku_revenue[sku],
        "Fixed Price": fixed_rev,
        "Elasticity Baseline": elastic_rev,
        "Random Baseline": random_rev
    })

baseline_df = pd.DataFrame(baseline_rows)
print("\nBaseline Comparison:\n", baseline_df)

#  AVERAGE TRAINING PERFORMANCE ACROSS SKUs

def value_iteration_with_logging(env):
    prices = env.price_grid
    nA = len(prices)

    V = np.zeros(nA)
    iter_vals, iter_deltas = [], []

    for iteration in range(MAX_ITER):
        delta = 0
        V_new = np.zeros_like(V)

        for i, p in enumerate(prices):
            rewards = np.zeros(nA)
            for j, p_next in enumerate(prices):
                rewards[j] = expected_reward(env, p_next) + GAMMA * V[j]

            best_idx = rewards.argmax()
            V_new[i] = rewards[best_idx]
            delta = max(delta, abs(V_new[i] - V[i]))

        V = V_new
        iter_vals.append(np.mean(V))
        iter_deltas.append(delta)

        if delta < BELLMAN_TOL:
            break

    return iter_vals, iter_deltas


print("\nComputing average training performance across SKUs...")
all_iter_vals = []

for sku in tqdm(candidate_skus[:50]):  # cap at 50 for speed
    sku_df = daily[daily['StockCode']==sku]
    env = DynamicPricingEnv(sku_df)
    vals, _ = value_iteration_with_logging(env)
    all_iter_vals.append(vals)

# Pad to same length
max_len = max(len(v) for v in all_iter_vals)
padded = [np.pad(v, (0, max_len - len(v)), constant_values=np.nan)
          for v in all_iter_vals]

avg_curve = np.nanmean(padded, axis=0)

plt.figure(figsize=(10,5))
plt.plot(avg_curve, linewidth=2)
plt.title("Average Value Function Convergence Across SKUs")
plt.xlabel("Iteration")
plt.ylabel("Mean V")
plt.grid()
plt.show()


#   Revenue Boxplot Across Baselines

comparison_df = pd.DataFrame({
    "RL": baseline_df["RL Revenue"],
    "Fixed Price": baseline_df["Fixed Price"],
    "Elasticity": baseline_df["Elasticity Baseline"],
    "Random": baseline_df["Random Baseline"]
})

plt.figure(figsize=(12,6))
sns.boxplot(data=comparison_df)
plt.title("Revenue Distribution Across SKUs – RL vs Baselines")
plt.ylabel("Simulated Revenue")
plt.grid(axis='y')
plt.show()
