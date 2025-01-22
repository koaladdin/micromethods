#= Problem Set 1 
1. Pull the course github repository from github.com
2. Create your own git repository (and publish to GitHub)
3. Write an algorithm in your language of choice to solve my Broyden example (not using a canned solver from a package in your language). 
4. Commit to your repository (repo) and push to GitHub. 
5. Submit your code and a screen shot of your repo to Canvas. 


using Plots
using Printf
using LinearAlgebra

#The Broyden example:
# System of equations
function f!(F, x)
    F[1] = x[1]^2 + x[2]^2 - 1
    F[2] = x[1] - x[2]
end

# Initial guess
x0 = [2.0, 1.0]

# Set up problem
prob = NonlinearProblem(f!, x0)


# Solve using Broyden
sol_broyden = solve(prob, Broyden(), abstol=1e-8)

println("\nBroyden solution: ", sol_broyden.u)
println("Broyden iterations: ", sol_broyden.iterations)




