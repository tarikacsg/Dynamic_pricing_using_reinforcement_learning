# Dynamic Pricing Using Reinforcement Learning

This project studies **dynamic pricing** as a sequential decision-making problem and formulates it using **reinforcement learning (RL)**. The goal is to learn pricing policies that adapt over time to uncertain and evolving consumer demand in order to maximize revenue.

Rather than relying on static or heuristic pricing rules, we model pricing as a **Markov Decision Process (MDP)** and apply **model-based reinforcement learning** to derive optimal pricing strategies under uncertainty.

---

## Problem Overview

In real-world retail and e-commerce systems, demand is:
- Stochastic and partially observable  
- Influenced by seasonality, promotions, and historical prices  
- Sensitive to price elasticity  

Incorrect pricing can lead to:
- Underpricing → lost revenue  
- Overpricing → reduced demand and customer churn  

Dynamic pricing addresses this by continuously updating prices based on observed demand signals. This project focuses on learning such policies using reinforcement learning rather than predefined rules.

---

## Dataset

We use the **Online Retail II Dataset** from the UCI Machine Learning Repository, which contains transactional data from a UK-based online retailer.

**Key characteristics:**
- Time period: 2009–2011  
- Transaction-level purchase data  
- Product prices, quantities, timestamps, and customer identifiers  

**Preprocessing steps include:**
- Removing returns and cancelled transactions  
- Filtering invalid prices and extreme outliers  
- Restricting analysis to UK transactions for market consistency  
- Aggregating data to daily SKU-level statistics to reduce noise  

---

## Modeling Approach

### Reinforcement Learning Formulation

The problem is modeled as an MDP with:

- **State**  
  - Current price  
  - Previous demand  
  - Inventory level (assumed infinite)  
  - Temporal features (e.g., day, period)

- **Action**  
  - Selecting the next price from a discretized price grid

- **Reward**  
  - Revenue: `R_t = p_t × q_t`

- **Transition Dynamics**  
  - Simulated using a **constant elasticity demand model**:  
    `q_t = A_t · p_t^{-ε} + ε_t`

### Solution Method

- Model-based RL  
- Value Iteration over discretized price actions  
- Transition probabilities estimated via repeated demand simulations  

This approach avoids the risks of online experimentation while allowing systematic policy optimization.

---

## Baselines

To contextualize RL performance, the learned policy is compared against:

- **Fixed Price Baseline** – historical median price  
- **Elasticity Baseline** – analytically optimal price under assumed demand model  
- **Random Baseline** – uniformly sampled prices  

---

## Results

- The RL policy outperforms historical pricing for the majority of evaluated SKUs  
- Significant revenue gains are observed for products with stable elasticity and sufficient price variation  
- Underperformance in some cases highlights sensitivity to demand-model assumptions  

Overall, results demonstrate that **dynamic adaptation over time** can outperform static optimal pricing in realistic settings.

---

## Key Takeaways

- Data quality and preprocessing are critical for stable RL behavior  
- Discretization simplifies learning while remaining effective  
- Demand model assumptions strongly influence outcomes  
- RL adds value beyond static pricing when demand evolves over time  

---

## Project Structure

