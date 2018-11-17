using Test
using Statistics
using CompositeConvex

m,n = 1000,100
num_noise = 10
A = randn(m,n)
x = randn(n)
b = A*x
e = zeros(m)
e[rand(1:m,num_noise)]  = 1000*randn(num_noise)
b_noisy = b + e

residual_norm = pNorm{1}()


c = (y,x)->affine_model!(y,A,x,b)
∇c_adjoint = (y,x,r)->mul!(y,A',r)
gauge_projection = (y,x,t)->compute_norm_projection!(y,x,t,residual_norm)
dual_gauge = x->compute_norm(x,dual_norm(residual_norm))
params = ProblemParams(c,∇c_adjoint,gauge_projection,dual_gauge)
info = IntermediateVars(zeros(m),zeros(m),0.0)

# Function evaluated at the noiseless solution should be zero
g = similar(x)
f = value_function_objective!(x, g, params, info)
@test f < 1e-10
@test norm(g) < 1e-10

f = value_function_objective!(zeros(n), g, params, info)
@test norm(g - (-A'*b)) < 1e-10

# Gradient test
info = IntermediateVars(zeros(m),zeros(m),1.0)
x0 = randn(n)
obj! = (x,g)->value_function_objective!(x,g,params,info)
f0 = obj!(x0,g)
dx = 1e-2*randn(n)
df = dot(g,dx)
h = 10 .^ (-6.0:0.0)
e1 = zeros(length(h))
for (i,hi) in enumerate(h)
    fi = obj!(x0+hi*dx,g)
    e1[i] = abs(fi - f0 - hi*df) 
end
taylor_exponent = median(diff(log10.(e1)))
@test abs(taylor_exponent - 2) < 0.1

# Recover the solution from noisy data, knowing the true τ value
c = (y,x)->affine_op!(y,A,x,b_noisy)
params = ProblemParams(c,∇c_adjoint,gauge_projection,dual_gauge)
info = IntermediateVars(zeros(m),zeros(m),norm(e,1))

obj! = (x,g)->value_function_objective!(x,g,params,info)
x1 = vmlmb(obj!, zeros(n), mem=10, ftol=(0,1e-3), maxiter=20, maxeval=100)
@test norm(x1-x) < 1e-3*norm(x)