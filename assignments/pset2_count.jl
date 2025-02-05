using CSV, DataFrames, Optim, StatsFuns

# Read the CSV file into a DataFrame
data = CSV.read("assignments/psychtoday.csv", DataFrame; header=["y", "x1", "x2", "x3", "x4", "x5", "x6"])

# Define the response and predictor variables
y = data.y
predictors = select(data, Not(:y))  # Ensure this excludes 'y'

# Convert predictors to a matrix
X = Matrix(predictors)

# Define the log-likelihood function for Poisson regression
function poisson_loglikelihood(β)
    λ = exp.(X * β) # Calculate the expected value λ for Poisson distribution
    ll = sum(y .* log.(λ) - λ - logfactorial.(y)) # Calculate log-likelihood
    return -ll  # Negative LL for minimization purposes
end

# Define the gradient of the log-likelihood function
function poisson_gradient!(gradient, β)
    λ = exp.(X * β) # Calculate the expected value λ for Poisson distribution
    gradient .= -X' * (y .- λ)  # Calculate gradient of log-likelihood
end

# Initial parameter vector (zeros)
initial_beta = zeros(size(X, 2))

# Helper function to run optimization and capture results
function run_optimization(algorithm_type, method_name; gradient=nothing)
    result, execution_time = if isnothing(gradient)
        start_time = time()
        result = optimize(poisson_loglikelihood, initial_beta, algorithm_type)
        (result, time() - start_time)
    else
        start_time = time()
        result = optimize(poisson_loglikelihood, gradient, initial_beta, algorithm_type; inplace=true)
        (result, time() - start_time)
    end

    estimated_beta = Optim.minimizer(result)
    num_iterations = result.iterations
    num_func_evals = result.f_calls

    return (method_name, estimated_beta, num_iterations, num_func_evals, execution_time)
end

# Define the BHHH optimization algorithm
function BHHH_with_hessian(X, y; tol=1e-6, maxiter=1000)
    β = zeros(size(X, 2))
    func_evals = 0

    initial_hessian = Matrix{Float64}(undef, size(X, 2), size(X, 2))
    final_hessian = Matrix{Float64}(undef, size(X, 2), size(X, 2))

    iteration_count = 0
    
    for i in 1:maxiter
        λ = exp.(X * β)
        score_matrix = X .* (y .- λ)[:, Colon()]
        func_evals += 1

        gradient = vec(sum(score_matrix, dims=1))
        hessian_approx = score_matrix' * score_matrix

        # Save the initial Hessian approximation
        if i == 1
            initial_hessian .= hessian_approx
        end

        if rank(hessian_approx) < size(hessian_approx, 1)
            println("Warning: Singular Matrix encountered; stopping early.")
            return (β, i, func_evals, initial_eigenvalues=[], final_eigenvalues=[])
        end

        step = pinv(hessian_approx) * gradient
        β_new = β + step

        if norm(β_new - β) < tol
            final_hessian .= hessian_approx
            iteration_count = i
            break
        end

        β = β_new
        final_hessian .= hessian_approx
        iteration_count = i
    end

    initial_eigenvalues = eigvals(initial_hessian)
    final_eigenvalues = eigvals(final_hessian)

    println("Initial Hessian Eigenvalues: ", initial_eigenvalues)
    println("Final Hessian Eigenvalues: ", final_eigenvalues)

    return (β, iteration_count, func_evals, initial_eigenvalues, final_eigenvalues)
end

# Perform optimizations using different methods
results = [
    run_optimization(BFGS(), "BFGS - Numerical"),
    run_optimization(BFGS(), "BFGS - Analytical", gradient=poisson_gradient!),
    run_optimization(NelderMead(), "Nelder-Mead"),
]

# Measure time and function evaluations for the BHHH optimization
start_time = time()
bhhh_beta, bhhh_iterations, bhhh_func_evals, initial_eigenvalues, final_eigenvalues = BHHH_with_hessian(X, y)
bhhh_time = time() - start_time

# Push BHHH results into the results array
push!(results, ("BHHH", bhhh_beta, bhhh_iterations, bhhh_func_evals, bhhh_time))

# Define the residual function for NLLS
function poisson_residual(β)
    λ = exp.(X * β)
    residuals = y .- λ
    return residuals
end

# Define the objective function for NLLS: sum of squared residuals
function nlls_objective(β)
    residuals = poisson_residual(β)
    return sum(residuals.^2)
end

# Perform NLLS optimization
function run_nlls_optimization(method_name)
    start_time = time()
    result = optimize(nlls_objective, initial_beta, BFGS())
    elapsed_time = time() - start_time

    estimated_beta = Optim.minimizer(result)
    num_iterations = result.iterations
    num_func_evals = result.f_calls

    return (method_name, estimated_beta, num_iterations, num_func_evals, elapsed_time)
end

# Run NLLS
nlls_results = run_nlls_optimization("NLLS")

# Add NLLS results to the overall results
push!(results, nlls_results)

# Create a table for results using all methods
results_table = DataFrame(
    Method = [r[1] for r in results],
    Estimates = [r[2] for r in results],
    Iterations = [r[3] for r in results],
    Function_Evals = [r[4] for r in results],
    Time = [r[5] for r in results]
)

println("All Optimization Results:")
println(results_table)

# Export the table to a LaTeX file
open("assignments/optimization_results_all.tex", "w") do io
    pretty_table(io, results_table, backend=Val(:latex))
end