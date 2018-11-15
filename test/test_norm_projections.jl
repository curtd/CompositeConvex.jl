using CompositeConvex
using LinearAlgebra

# Testing 1D norms
n = 100
x = randn(n)

τ0 = 2
for p in [0,1,2]
    y = compute_norm_projection(x, τ0, pNorm{p}())
    @test abs(norm(y, p) - τ0) < 1e-2
end

# Testing 2D norms
m,n = 100,100
x = randn(m,n)
y = similar(x)
row_norms = t->reshape(sum(t.^2,dims=2).^(0.5),size(t,2))
col_norms = t->reshape(sum(t.^2,dims=1).^(0.5),size(t,1))
compute_norm_projection!(y,x,τ0,pqNorm{0,2}())
@test abs(norm(col_norms(y),0) - τ0) < 1e-2

compute_norm_projection!(y,x,τ0,pqNorm{1,2}())
@test abs(norm(col_norms(y),1) - τ0) < 1e-2