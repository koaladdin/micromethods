# Examples

These are two examples. They both estimate a nonlinear least-squares type objective function, and also need to solve a system on nonlinear equation for each estimation iteration. 

1. Anderson & Van Wincoop (2003 AER ) -- "Gravity with Gravitas"
2. Berry, Levinsohn, and Pakes (1995 Econometrica) -- "Automobile Prices in Market Equilibrium"

## Gravity in International Trade
**Anderson & Van Wincoop (2003 AER ) - Gravity with Gravitas**


#### Core Contribution
- Previous work found a "border puzzle:" Canadian provinces trade 22 times more with each other than with US states. The authors show this estimate is biased due to omitted multilateral resistance terms and develop a theoretically-grounded gravity equation.

E.g. MacCallum (1995) estimated:
$$ln(x_{ij}) = α_1 + α_2 log(y) + α_3 log(y_j) + α_4 log(d_{ij}) α_5 δ_{ij} +ε_{ij}$$


#### Theoretical Framework
- Derives a "gravity equation" from first principles using CES preferences and trade costs
- Key insight: Trade between two regions depends on bilateral trade barriers relative to average trade barriers with all trading partners ("multilateral resistance")
- Trade flow equation:
  $$x_{ij} = (y_i * y_j/y_w) * (t_{ij}/P_i*P_j)^{(1-σ)}$$
  where:
  - $x_ij$ is exports from region i to j
  - $y_i$, $y_j$ are incomes
  - $y_w$ is world income
  - $t_{ij}$ is bilateral trade cost
  - $P_i$, $P_j$ are multilateral resistance terms
  - $σ$ is elasticity of substitution

#### Empirical Implementation
- Uses US-Canada trade data (1993)
- Estimates border effect using:
  1. Traditional gravity equation (replicating McCallum)
  2. Theory-consistent approach with multilateral resistance
- Develops custom nonlinear estimation method due to $P_i$ terms being unobservable
- Controls for distance, borders, and other bilateral trade costs

#### Key computational problem

The $P_i$ terms are not observed in the data, but they are implied by the CES model used to derive the trade flow equation. 

**Solution:** For each guess of the parameters, solve the model to find $P_i$,$P_j$, and then estimate the trade flow ($x_{ij}$) relationship. 

#### Equilibrium condition for resistance terms ($P_i$)

Price Index Equations
The multilateral resistance terms ($P_i$ and $P_j$) are derived as CES price indices:
$$P_i^{(1-σ)} = \sum_{j=1..J} θ_j * (t_{ij}/P_j)^{(1-σ)}$$
where:

- $θ_j$ is region j's share of world income $(y_j/y_w)$
- $t_{ij}$ represents bilateral trade costs
- $σ$ is the elasticity of substitution
- $j$ indexes all coutnries.
  
This equation must hold for all $i$ regions simultaneously, so if there are 20 countires, we have a system of 20 nonlinear equations and 20 unknowns. 

#### Trade Cost Function
The bilateral trade costs $(t_ij)$ are modeled as:
$$
ln(t_{ij}) = b + ρ*ln(d_{ij}) + δ*(1-B_{ij})
$$
where:
- $d_{ij}$ is distance between regions
- $B_{ij}$ is border dummy (1 if same country)
- $b$ is a constant
- $ρ$ captures distance effect
- $δ$ measures border effect

#### Steps in the Algorithm:

1. **Initial Setup**
   - Start with initial guess for parameters ${b, ρ, δ, σ}$
   - Calculate initial trade costs $t_{ij}$ using trade cost function

2. **Solve for P_i (Fixed Point Iteration)**
   - Start with initial guess for all $P_i = 1$
   - Iteratively update $P_i$ using price index equation until convergence
   - For each iteration:
    $$
    \vec{P}^{new} = \big[ \sum_j (θ_j * (t_{ij}/\vec{P}^{old})^{(1-σ)}) \big]^{1/(1-σ)}
    $$
   - Continue until $\max \mid \vec{P}^{new} - \vec{P}^{old} \mid < tol$

3. **Calculate Predicted Trade Flows**
   $$
   \hat{x}_{ij} = (y_i * y_j/y_w) * (t_{ij}/P_i*P_j)^{(1-σ)}
   $$

4. **Compute Objective Function**
   - Use nonlinear least squares:
   $$
   S(θ) = \sum_{ij} \Big[ln(x_{ij}^{data}) - ln(\hat{x}_{ij})\Big]^2
   $$

5. **Update Parameters**
   - Use numerical optimization (custom Newton method) to update ${b, ρ, δ}$
   - Return to step 2 with new parameters
   - Continue until convergence





## Differentiated Products Demand
*(this is a simplified version of their paper)* 

Berry, Levinsohn, Pakes (1995 ECMA) propose a way to estimate discrete choice demand for differentiated products 
- with product-level data (shares, prices, product characteristics) 
- incorporating endogenous prices
- random effect on consumer preferences for product characteristics

Contribution: a very flexible model (in terms of substitution patterns) recoverd from widely available data. 

#### Choice Model
(This is a very reduced version of this model)

Consumer $i$ has indirect utility for product $j$ that is a function of product characteristics $(x,ξ)$, price $(p)$, and a random match term $ε$: 
$$
u_{ijt} = x_{jt}\beta + ν σ_i x_{jt} - α p_{jt}  + ξ_{jt} + ε_{ijt}
$$
- $β, ν, α$ are preference parameters
- $ξ$ is unobserved
- $σ_i$ is a consumer level unobserved term

Consumers make a discrete choice among all the products in $j \in \mathcal{J}$ indexed $j=1..J$ and the outside option that has utility $u_{i0t}=ε_{i0t}$.

Choice probabilities are 
$$
s_{ijt} = \frac{exp(\delta_{jt} + ν σ_i x_{jt})}{1 + \sum_{k\in\mathcal{J}} exp(\delta_{kt} + ν σ_i x_{kt})}
$$
where 
$$
\delta_{jt} = x_{jt}\beta - α p_{jt}  + ξ_{jt}
$$

Aggregate choice probabilities (or market shares) are 

$$
s_{jt} = \int\frac{exp(\delta_{jt} + ν σ_i x_{jt})}{1 + \sum_{k\in\mathcal{J}} exp(\delta_{kt} + ν σ_i x_{kt})} dG(\sigma)
$$

#### Estimation

BLP propose estiamting the parameters $θ = (β,α,ν)$ using the following restriction on the distribution of $\xi$.
$$
E\big[\xi \mid Z \big] = 0
$$
for some instruments $z_{jt}\in Z$ 

In practice, we turn this into an unconditional moment equality 
$$
E\big[\xi' * Z\big] = 0
$$

**Computational Problem**

On one hand, this makes sense because the $\xi$ term looks like a residual of the $\delta$ equation, which itself is linear in some of the parameters.  

On the other hand, $\delta$ is not observed, and it is sort of buried in the model. 

**Computational Solution**

BLP propose solve for $\theta$ with the following system of non-linear equations. 
$$
s^{data} = s_{jt}(δ;θ)
$$
1. This turns out to have a unique solution (Berry 1994, RAND). 
2. $δ$ can be found with a contraction mapping (function iteration) method.
$$
δ^{new} = δ^{old} + log(s^{data}) - log(s_{jt}(δ^{old};θ))
$$

#### BLP Estimation Procedure

1. Initial guess of $ν^0$
2. Find delta using contraction mapping: $\hat{δ}(θ^0)$.
3. Project $\hat{δ}(θ^0)$ on  $x_{jt}\beta - α p_{jt}$ to get implied residual $\hat{ξ}$.
4. Form GMM moment $$(\xi' * Z)' W (\xi' * Z)$$
5. Minimize this moment vector by repeating with updated guesses of $\nu$ and go back to Step 2.

(note: this is a sort of inner-outer structure, but can also be treated a single step)