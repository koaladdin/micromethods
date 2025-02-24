using Distributions
using LinearAlgebra
using Statistics
using Plots
using JLD2

# Function to compute log-likelihood (assumed to be defined elsewhere)
function logLike(θ, data)
    μ = θ[1]
    σ = abs(θ[2])
    d = Normal(μ, σ)
    return sum(logpdf.(d, data))
end

# Load or generate data
# Uncomment to generate new data
# μ_a = 2
# var_a = 2
# cases = 500
# data_n24 = rand(Normal(μ_a, sqrt(var_a)), cases)
# @save "data_n24.jld2" data_n24

# Load existing data
@load "data_n24.jld2" data_n24
data = data_n24

# Prior distribution parameters
prsig = zeros(2, 2)
prsig[1,1] = 500
prsig[2,2] = 500
prmu = [10.0, 2.0]

# Proposal distribution parameters
promu = zeros(2)
prosig = 0.1 * I(2)  # I creates an identity matrix

# Initial values
θ₀ = fill(5.0, 2)

# Objective functions
objfnc(x) = logLike(x, data)
objfncSA(x) = -logLike(x, data)

# Metropolis-Hastings implementation
println("HANDWRITTEN MH TEST")
@time begin
    T = 20000  # total length of chain
    θ = zeros(T, 2)
    θ[1,:] = θ₀
    
    # Initial log posterior
    curr_pi = objfnc(θ₀) + logpdf(MvNormal(prmu, prsig), θ₀)
    
    iaccept = 0  # acceptance counter
    
    for t in 2:T
        # Generate proposal
        innov = promu + rand(MvNormal(zeros(2), prosig))
        propose = θ[t-1,:] + innov
        propose[2] = abs(propose[2])
        
        # Compute log posterior at proposal
        lik = objfnc(propose)
        prior = logpdf(MvNormal(prmu, prsig), propose)
        prop_pi = lik + prior
        
        # Compute acceptance probability
        delta = prop_pi - curr_pi
        accprob = min(0, delta)
        
        # Accept/reject step
        if log(rand()) < accprob
            iaccept += 1
            θ[t,:] = propose
            curr_pi = prop_pi
        else
            θ[t,:] = θ[t-1,:]
        end
    end
    
    # Save results
    @save "theta.jld2" θ
    
    # Compute summary statistics
    B = 500  # burn-in period
    mtheta = mean(θ[B+1:end, :], dims=1)
    vtheta = var(θ[B+1:end, :], dims=1)
    acc_rate = iaccept/T
    
    # Plot results
    p = plot(θ[1:1000,1], title="Time Series Plot of First Thousand Draws of mu",
             xlabel="Iteration", ylabel="μ")
    display(p)
    savefig(p, "mh_plot.png")
    
    # Write output to file
    open("metroHastMain.out", "w") do f
        println(f, "\nSUMMARY STATISTICS")
        println(f, "---------------------------------------------------------\n")
        println(f, "Mean of posterior mu distribution:            $(mtheta[1])")
        println(f, "Mean of posterior sigma distribution:         $(mtheta[2])")
        println(f, "Variance of posterior mu distribution:        $(vtheta[1])")
        println(f, "Variance of posterior sigma distribution:     $(vtheta[2])")
        println(f, "Acceptance rate:                             $acc_rate")
    end
end

# Using built-in MCMC sampler (if available)
println("\nMCMC with Turing.jl (alternative to MATLAB's mhsample)")
println("Note: For a direct equivalent to MATLAB's mhsample, consider using the MCMCChains.jl package")

# Simulated Annealing implementation
println("\nSimulated Annealing")
# Note: Julia has packages like OptimizationMetaheuristics.jl that provide 
# simulated annealing functionality. The exact equivalent would require
# additional configuration.