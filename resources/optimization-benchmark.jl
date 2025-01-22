using Optim
using BenchmarkTools
using ForwardDiff
using Statistics
using DataFrames
using Plots
using Printf
using LinearAlgebra

# Golden Section Search implementation
function golden_section_1d(f, a, b, tol=1e-8, max_iter=1000)
    φ = (1 + √5) / 2  # golden ratio
    ρ = 1/φ
    d = ρ * (b - a)
    x1 = b - d
    x2 = a + d
    f1 = f(x1)
    f2 = f(x2)
    
    for _ in 1:max_iter
        if f1 < f2
            b = x2
            x2 = x1
            f2 = f1
            d = ρ * (b - a)
            x1 = b - d
            f1 = f(x1)
        else
            a = x1
            x1 = x2
            f1 = f2
            d = ρ * (b - a)
            x2 = a + d
            f2 = f(x2)
        end
        
        if abs(b - a) < tol
            return (a + b) / 2
        end
    end
    
    return (a + b) / 2
end

"""
2D Golden Section Search using coordinate descent
"""
function golden_section_2d(f, x0, bounds, tol=1e-8, max_iter=1000)
    x = copy(x0)
    x_history = [copy(x)]
    iter = 0
    
    for _ in 1:max_iter
        x_old = copy(x)
        
        # Optimize along x1 direction
        f1 = t -> f([t, x[2]])
        x[1] = golden_section_1d(f1, bounds[1][1], bounds[1][2], tol)
        
        # Optimize along x2 direction
        f2 = t -> f([x[1], t])
        x[2] = golden_section_1d(f2, bounds[2][1], bounds[2][2], tol)
        
        push!(x_history, copy(x))
        iter += 1
        
        # Check convergence
        if norm(x - x_old) < tol
            break
        end
    end
    
    return x, x_history, iter
end

# Wrapper for optimization interface
function optimize_golden_section(f, x0, bounds; 
                               tol=1e-8, max_iter=1000)
    x_min, history, iter = golden_section_2d(f, x0, bounds, tol, max_iter)
    converged = iter < max_iter
    
    return (minimizer=x_min, 
            minimum=f(x_min), 
            iterations=iter, 
            converged=converged)
end

"""
Run multiple trials of optimization with different starting points
"""
function benchmark_from_multiple_starts(f, g!, method, starts; max_time=5.0)
    results = []
    
    for x0 in starts
        # Set up optimization parameters
        opts = Optim.Options(
            iterations=1000,
            time_limit=max_time,
            show_trace=false
        )
        
        # Time the optimization
        if method isa NelderMead
            time = @elapsed result = optimize(f, x0, method, opts)
        else
            time = @elapsed result = optimize(f, g!, x0, method, opts)
        end
        
        push!(results, (
            time=time,
            iterations=Optim.iterations(result),
            minimum=Optim.minimum(result),
            converged=Optim.converged(result),
            x_final=Optim.minimizer(result)
        ))
    end
    
    return results
end

"""
Generate statistics from benchmark results
"""
function analyze_results(results)
    times = [r.time for r in results]
    iterations = [r.iterations for r in results]
    minima = [r.minimum for r in results]
    converged = [r.converged for r in results]
    
    return (
        mean_time=mean(times),
        std_time=std(times),
        mean_iterations=mean(iterations),
        std_iterations=std(iterations),
        best_minimum=minimum(minima),
        worst_minimum=maximum(minima),
        convergence_rate=mean(converged),
        n_trials=length(results)
    )
end

"""
Create a visualization of optimization paths
"""
function plot_optimization_paths(f, bounds, results; resolution=100, func_name="", method_name="")
    x = range(bounds[1][1], bounds[1][2], length=resolution)
    y = range(bounds[2][1], bounds[2][2], length=resolution)
    z = [f([i, j]) for i in x, j in y]
    
    # Create main title with function and method information
    n_starts = length(results)
    main_title = "Optimization of $(func_name) Function\n" *
                 "Method: $(method_name)\n" *
                 "Number of Different Starting Points: $(n_starts)"
    
    # Create plot with enhanced information
    p = contour(x, y, z, 
                fill=true, 
                title=main_title,
                xlabel="x₁",
                ylabel="x₂",
                colorbar_title="Function Value",
                titlefont=font(10),
                margin=10Plots.mm)
    
    # Plot final points from all trials
    final_points = [r.x_final for r in results]
    xs = [p[1] for p in final_points]
    ys = [p[2] for p in final_points]
    scatter!(p, xs, ys, 
             label="Final Points",
             color=:red,
             markersize=4,
             markershape=:circle)
    
    # Add convergence information
    n_converged = sum([r.converged for r in results])
    convergence_rate = round(100 * n_converged / n_starts, digits=1)
    annotate!(p, bounds[1][1], bounds[2][2], 
             text("Convergence Rate: $(convergence_rate)%", :left, 8))
    
    return p
end

# Test functions
# Rosenbrock (Banana) function
banana(x) = 100 * (x[2] - x[1]^2)^2 + (1 - x[1])^2

function banana_gradient!(g, x)
    g[1] = -400 * x[1] * (x[2] - x[1]^2) - 2 * (1 - x[1])
    g[2] = 200 * (x[2] - x[1]^2)
end

# Himmelblau function
himmelblau(x) = (x[1]^2 + x[2] - 11)^2 + (x[1] + x[2]^2 - 7)^2

function himmelblau_gradient!(g, x)
    # First term derivatives: (x^2 + y - 11)^2
    term1 = x[1]^2 + x[2] - 11
    dx1 = 2 * term1 * (2*x[1])  # chain rule for x
    dy1 = 2 * term1 * (1)       # chain rule for y
    
    # Second term derivatives: (x + y^2 - 7)^2
    term2 = x[1] + x[2]^2 - 7
    dx2 = 2 * term2 * (1)       # chain rule for x
    dy2 = 2 * term2 * (2*x[2])  # chain rule for y
    
    g[1] = dx1 + dx2
    g[2] = dy1 + dy2
end

# Run benchmarks
function run_full_benchmark()
    # Test functions to benchmark
    functions = [
        (f=banana, g=banana_gradient!, name="Rosenbrock"),
        (f=himmelblau, g=himmelblau_gradient!, name="Himmelblau")
    ]
    
    # Define optimization methods and their gradient approaches
    method_configs = [
        (method=NelderMead(), name="NelderMead", grad_type="None"),
        (method=BFGS(), name="BFGS", grad_type="Analytical"),
        (method=BFGS(), name="BFGS-FD", grad_type="Numerical"),
        (method=BFGS(), name="BFGS-AD", grad_type="AutoDiff"),
        (method=GradientDescent(), name="GradientDescent", grad_type="Analytical"),
        (method=ConjugateGradient(), name="ConjugateGradient", grad_type="Analytical"),
        (method="GoldenSection", name="GoldenSection", grad_type="None")
    ]
    
    # Generate random starting points
    n_trials = 10
    starts = [2 .* randn(2) for _ in 1:n_trials]
    
    # Store results
    all_results = DataFrame(
        function_name=String[],
        method_name=String[],
        gradient_type=String[],
        mean_time=Float64[],
        std_time=Float64[],
        mean_iterations=Float64[],
        convergence_rate=Float64[],
        best_minimum=Float64[]
    )
    
    for f in functions
        bounds = ((-4.0, 4.0), (-4.0, 4.0))  # Search bounds
        
        for config in method_configs
            # Select appropriate gradient function based on type
            gradient_func = if config.grad_type == "None"
                nothing
            elseif config.grad_type == "Numerical"
                # Simple forward differences
                (g, x) -> begin
                    h = 1e-8  # Step size
                    for i in 1:length(x)
                        x_plus_h = copy(x)
                        x_plus_h[i] += h
                        g[i] = (f.f(x_plus_h) - f.f(x)) / h
                    end
                end
            elseif config.grad_type == "AutoDiff"
                (g, x) -> ForwardDiff.gradient!(g, f.f, x)
            else  # Analytical
                f.g
            end
            
            # Run optimization
            if config.method == "GoldenSection"
                results = []
                bounds = ((-4.0, 4.0), (-4.0, 4.0))
                
                for x0 in starts
                    time = @elapsed result = optimize_golden_section(f.f, x0, bounds)
                    push!(results, (
                        time=time,
                        iterations=result.iterations,
                        minimum=result.minimum,
                        converged=result.converged,
                        x_final=result.minimizer
                    ))
                end
            else
                results = benchmark_from_multiple_starts(
                    f.f, gradient_func, config.method, starts
                )
            end
            
            stats = analyze_results(results)
            
            push!(all_results, (
                f.name,
                config.name,
                config.grad_type,
                stats.mean_time,
                stats.std_time,
                stats.mean_iterations,
                stats.convergence_rate,
                stats.best_minimum
            ))
            
            # Generate visualization
            p = plot_optimization_paths(f.f, bounds, results,
                                     func_name=f.name,
                                     method_name="$(config.name)\n($(config.grad_type) Gradients)")
            display(p)
            savefig(p, "$(f.name)_$(config.name)_$(config.grad_type)_paths.png")
        end
    end
    
    # Generate Markdown table
    println("\nMarkdown Table:")
    println("| Function | Method | Gradient Type | Mean Time (s) | Std Time | Mean Iterations | Convergence Rate | Best Minimum |")
    println("|----------|---------|---------------|--------------|-----------|-----------------|-----------------|--------------|")
    
    for row in eachrow(all_results)
        formatted_row = "| $(row.function_name) | $(row.method_name) | $(row.gradient_type) | " *
                       "$(round(row.mean_time, digits=4)) | $(round(row.std_time, digits=4)) | " *
                       "$(round(row.mean_iterations, digits=1)) | $(row.convergence_rate) | " *
                       "$(@sprintf("%.2e", row.best_minimum)) |"
        println(formatted_row)
    end
    
    return all_results
end

# Run the full benchmark and display results
results = run_full_benchmark()
println("\nBenchmark Results:")
println(results)