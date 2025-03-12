using Random
using Distributions
using CSV
using DataFrames
using Optim

## {Part 3: Contraction Mapping}

# Parameters
μ = -1.0
R = -3.0
β = 0.9
γ = 0.5775 # Euler's constant
tolerance = 1e-6
max_iter = 1000

# State space for machine ages
ages = 1:5

# Initialize value function
V = zeros(length(ages))

# Function to calculate choice-specific value
function choice_specific_value(a, θ, V, is_replace)
    if is_replace
        return θ[2] + β * V[1] # Replacement: a_t+1 = 1
    else
        next_age = min(a + 1, 5)
        return θ[1] * a + β * V[next_age] # Non-replacement
    end
end

# Value function iteration
function value_iteration()
    for iter in 1:max_iter
        V_new = similar(V)
        
        for a in ages
            V0 = choice_specific_value(a, (μ, R), V, false)
            V1 = choice_specific_value(a, (μ, R), V, true)

            # Log-Sum-Exp formula
            V_new[a] = log(exp(V0) + exp(V1)) + γ
        end
        
        # Check for convergence
        if norm(V_new - V) < tolerance
            println("Converged after $iter iterations.")
            return V_new
        end
        
        V .= V_new
    end
    
    error("Value function iteration did not converge.")
end

# Run value function iteration
V_star = value_iteration()

# Print the final value function
println("Optimal Value Function: ", V_star)

# Function to determine epsilon difference for indifference
function indifference_epsilon_diff(age, V_star)
    next_age_non_replace = min(age + 1, 5)
    ε_diff = -1 + β * (V_star[1] - V_star[next_age_non_replace])
    return ε_diff
end

# Calculate indifference for age 2
ε_diff = indifference_epsilon_diff(2, V_star)
println("Epsilon difference for indifference at age 2: ", ε_diff)

# Determine replacement probability for a given age and value function state
function replacement_probability(age, V_star)
    V0 = choice_specific_value(age, (μ, R), V_star, false)
    V1 = choice_specific_value(age, (μ, R), V_star, true)
    
    prob_replace = exp(V1) / (exp(V0) + exp(V1))
    return prob_replace
end

# Calculate the probability of replacement when age is 2
prob_replace = replacement_probability(2, V_star)
println("Probability of replacement at age 2: ", prob_replace)

# Compute the value at specific state with given shocks
function specific_state_value(age, ε0, ε1, V_star)
    V0 = μ * age + ε0 + β * V_star[min(age + 1, 5)]
    V1 = R + ε1 + β * V_star[1]
    log(exp(V0) + exp(V1)) + γ
end

# Calculate value for age 4, ε0 = 1, ε1 = 1.5
value_at_specific_state = specific_state_value(4, 1.0, 1.5, V_star)
println("Value at state (a_t = 4, ε0t = 1, ε1t = 1.5): ", value_at_specific_state)

## {Part 4: Simulate Data}

# Function to generate synthetic data and save as CSV
function generate_data_to_csv(T, V_star, filename)
    μ, R, β = -1.0, -3.0, 0.9
    a_states, i_choices = zeros(Int, T), zeros(Int, T)
    a_states[1] = 1
    ϵ_distribution = Gumbel(0, 1)

    for t in 1:T
        a_t = a_states[t]
        ε0t, ε1t = rand(ϵ_distribution), rand(ϵ_distribution)
        V0 = μ * a_t + ε0t + β * V_star[min(a_t + 1, 5)]
        V1 = R + ε1t + β * V_star[1]
        p_replace = exp(V1) / (exp(V0) + exp(V1))
        i_t = rand() < p_replace ? 1 : 0
        i_choices[t] = i_t
        if t < T
            a_states[t + 1] = i_t == 1 ? 1 : min(a_t + 1, 5)
        end
    end

    df = DataFrame(age_state = a_states, choice = i_choices)
    CSV.write(filename, df)
    println("Data written to ", filename)
end

# Generate data for T = 20,000 periods and save to CSV
T = 20_000
generate_data_to_csv(T, V_star, "machine_data.csv")

## {Part 5: Rust NFP Estimation}

# Read in simulated data
df = CSV.read("machine_data.csv", DataFrame)
states = df.age_state
choices = df.choice

# Logsum function
logsum(ε0, ε1) = log(exp(ε0) + exp(ε1)) + γ

# Function to solve for conditional value functions V0 and V1 using contraction mapping
function solve_value_functions(θ, β, V)
    μ, R = θ
    V_new = zeros(5)  # Initialize value function array
    converged = false
    tolerance = 1e-6
    max_iterations = 1000
    iteration = 0
    
    while !converged && iteration < max_iterations
        iteration += 1
        V_temp = copy(V_new)
        
        for a_t in 1:5
            # For action 0 (not replacing)
            if a_t < 5
                Vbar_0 = μ * a_t + β * V[a_t + 1] 
            else
                Vbar_0 = μ * a_t + β * V[5]
            end

            # For action 1 (replacing)
            Vbar_1 = R + β * V[1]
            
            # Compute the conditional value functions
            V_new[a_t] = logsum(Vbar_0, Vbar_1)  # Logsum computation
            
        end
        
        # Check for convergence
        if maximum(abs.(V_new - V_temp)) < tolerance
            converged = true
        end
    end
    println("V_new is ", V_new )
    return V_new
end

# Function to compute the likelihood
function likelihood(θ, states, choices, β, V)
    μ, R = θ
    log_like = 0.0
    
    # Solve for the conditional value functions for the current θ
    V_new = solve_value_functions(θ, β, V)
    
    for t in 1:length(states)
        a_t = states[t]
        choice = choices[t]
        
        # Compute the values for the two actions
        if a_t < 5
            Vbar_0 = μ * a_t + β * V_new[a_t + 1] 
        else
            Vbar_0 = μ * a_t + β * V_new[5]
        end
        Vbar_1 = R + β * V_new[1]
        
        # Compute the logit probability of choosing to replace
        P_replace = exp(Vbar_1) / (exp(Vbar_0) + exp(Vbar_1))
        
        # Add the log-likelihood contribution for this observation
        if choice == 1
            log_like += log(P_replace)
        else
            log_like += log(1 - P_replace)
        end
    end
    return -log_like  # Return negative log-likelihood for minimization
end

# Initial guess for θ = (μ, R)
initial_guess = [0.0, 0.0]

# Use optimization to maximize the likelihood function (minimize negative log-likelihood)
result = optimize(θ -> likelihood(θ, states, choices, β, zeros(5)), initial_guess, NelderMead())

# Get the estimated θ
estimated_θ = result.minimizer
println("Estimated θ (NFP) = ", estimated_θ)

## {Part 6: Holtz Miller CCP estimation}

# Estimate replacement probabilities at each state using the average replacement rate
function estimate_replacement_probabilities(states, choices)
    P_hat = zeros(5)  # Store the estimated replacement probabilities for each state
    
    for a_t in 1:5
        # Get the indices where the machine is in state a_t
        indices = findall(states .== a_t)
        
        # Count the number of replacements and total observations for a_t
        num_replacements = sum(choices[indices] .== 1)
        total_observations = length(indices)
        
        # Compute the estimated replacement probability for state a_t
        P_hat[a_t] = num_replacements / total_observations
    end
    
    return P_hat
end

# Compute the replacement probabilities
P_hat = estimate_replacement_probabilities(states, choices)
println("Estimated replacement probabilities: ", P_hat)

# Transition matrix for not replacing the machine (F_0)
function create_F0()
    F0 = zeros(5, 5)
    for a_t in 1:4
        F0[a_t, a_t + 1] = 1  # Machine ages by 1
    end
    F0[5, 5] = 1  # Machine stays at age 5
    return F0
end

# Transition matrix for replacing the machine (F_1)
function create_F1()
    F1 = zeros(5, 5)
    for a_t in 1:5
        F1[a_t, 1] = 1  # Machine always resets to age 1 when replaced
    end
    return F1
end

# Generate the transition matrices
F_0 = create_F0()
F_1 = create_F1()

# Function to perform forward simulation
function forward_simulation(P_hat, F_0, F_1, β, V, μ, R)
    # Initialize the value functions for each state
    V0 = zeros(5)
    V1 = zeros(5)
    
    for a_t in 1:5
        # Compute the expected value for not replacing (V0) and using the state transition matrix F_0
        V0[a_t] = μ * a_t + β * sum(F_0[a_t, :] .* V)
        
        # Compute the expected value for replacing (V1) and using the state transition matrix F_1
        V1[a_t] = R + β * sum(F_1[a_t, :] .* V)
        
        # Weight the two value functions by the replacement probability P(a_t)
        # This gives the expected value depending on whether the machine is replaced or not
        V0[a_t] = (1 - P_hat[a_t]) * V0[a_t]  # Not replacing
        V1[a_t] = P_hat[a_t] * V1[a_t]        # Replacing
    end
    
    return V0, V1
end

# Function to compute the likelihood for CCP approach
function likelihood_ccp(θ, β, V, states, choices, P_hat, F_0, F_1)
    μ, R = θ
    log_like = 0.0
    
    # Solve for the conditional value functions for the current θ
    Vbar_0, Vbar_1 = forward_simulation(P_hat, F_0, F_1, β, V, μ, R)
    
    # Loop over the dataset to compute the log-likelihood
    for t in 1:length(states)
        a_t = states[t]
        choice = choices[t]
        
        # Compute the logit probability of choosing to replace
        P_replace = exp(Vbar_1[a_t]) / (exp(Vbar_0[a_t]) + exp(Vbar_1[a_t]))
        
        # Add the log-likelihood contribution for this observation
        if choice == 1
            log_like += log(P_replace)
        else
            log_like += log(1 - P_replace)
        end
    end
    return -log_like  # Return negative log-likelihood for minimization
end

# Initial guess for θ = (μ, R)
initial_guess = [-1.0, -3.0]

# Use optimization to maximize the likelihood function (minimize negative log-likelihood)
result = optimize(θ -> likelihood_ccp(θ, β, zeros(5), states, choices, P_hat, F_0, F_1), initial_guess, NelderMead())

# Get the estimated θ
estimated_θ = result.minimizer
println("Estimated θ (CCP) = ", estimated_θ)