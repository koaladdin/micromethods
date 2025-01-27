# Elementary Methods

We will talk abotu two "heuristics" that can sometimes be very helpful for solving problems. 


### Gauss-Jacobi

Let's say we have a system of N nonlinear equations and N unknowns. 

$$
\begin{align*}
f_1(x_1,x_2,...,x_N) = 0 \\ 
f_2(x_1,x_2,...,x_N) = 0 \\
... \\
f_N(x_1,x_2,...,x_N) = 0
\end{align*}
$$

Also, order the $x$'s so that
$$
J(x) = 
\begin{pmatrix}
\uparrow & . & . \\
. & \uparrow & . \\
. & . & \uparrow.
\end{pmatrix}
$$
In other words, the $x$ that affects $f_i$ the most in the in diagonal position. 

What if $J(x)$ is a nightmare to compute, but $f'_i(x_i,x_{-i})$ is not that bad?

Method: Go through the system of equations, one equation at a time. Solve each equation for the dominant unknown, conditional on the other unknowns. 

**Gauss-Jacobi Method**

1. Guess $x^0$
2. Solve $f_1(x_1;x)=0$ and find and save $x^0_1\rightarrow x^1_1$
2. Do that for each sub-equation, keeping $x^0$.
3. After all of the equations, update vector of $x^0 \rightarrow x^1$.
4. Repeat


### Gauss Seidel-Method

Similar to Gauss-Jacobi, but update $x_i^k$ after each sub-solution. 

1. Solve $f_1(x^0_1;x^0_2,x^0_3,...x^0_N)=0$
2. Solve $f_1(x^0_2;x^1_1,x^0_3,...x^0_N)=0$
3. Keep replacing for each sub-problem.

### Notes

- GS obviously depends on ordering and GJ does not. 
- The problem needs this "diagonally dominant" structure.
- A problem that works well is a Nash-pricing game w/ differentiated products. My price depends mostly on my elasticity of demand, and only a little on rivals cross-elasticities.

