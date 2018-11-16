module Norms

# Module containing structures for computing various matrix/vector norms, 
# their duals, projections onto their corresponding level sets

using LinearAlgebra

export pNorm, pqNorm, SchattenNorm, dual_norm, compute_norm, compute_norm_projection

squeeze(x) = reshape(x,size(x,1))
validate_p_value(P) = (isa(P,Real) && P >= 0) || (isa(P,Float64) && P==Inf) || error("Invalid value of P: $(P)")

abstract type Norm end 
abstract type VectorNorm <: Norm end
abstract type MatrixNorm <: Norm end
struct pNorm{P} <: VectorNorm 
    function pNorm{P}() where {P}
        validate_p_value(P)
        new()
    end
end

"""
dual_norm(p::pNorm{P}) -> pNorm{Q}

Computes the dual norm corresponding to a given vector p-norm, i.e., the value ``Q`` such that 

``\\dfrac{1}{P} + \\dfrac{1}{Q} = 1``

``P`` should be such that ``0 < P ≤ ∞.``
"""

dual_norm(p::pNorm{P}) where {P} = pNorm{1/(1-1/P)}()
dual_norm(p::pNorm{2}) = pNorm{2}()
dual_norm(p::pNorm{1}) = pNorm{Inf}()
dual_norm(p::pNorm{Inf}) = pNorm{1}()
dual_norm(p::pNorm{0}) = error("The 0-norm is not a true norm and does not have a corresponding dual")


"""
compute_norm(v::Vector, p::pNorm{P}) -> Float64

Computes the P-norm of v
"""
compute_norm(v::Vector, p::pNorm{P}) where {P} = LinearAlgebra.norm(v,P)

"""
    compute_norm_projection(x::Vector, τ, p::pNorm{P}) -> y::Vector 
    
    Variant of compute_norm_projection! that produces an output vector to store the projected vector

    Computes the projection of the vector `x` onto level set of the p-norm specified by P, of size τ, i.e., solves the problem 

    ``
        \\min_{y} & 0.5*\\|y - x\\|_2^2 \\\\
        \\text{s.t.} & \\; \\|y\\|_P \\le \\tau
    ``

    P can be one of 0, 1, 2, Inf or any real number with 0 < P < Inf. 

    # Example 
    ```jldoctest
    julia> x = randn(5);
    5-element Array{Float64,1}:
     0.9270834427394729
     0.5417380391296652
     1.264305123889455
     0.021179212004242474
     2.002779212447857

    julia> y = compute_norm_projection(x, 1, pNorm{1}())
    5-element Array{Float64,1}:
     0.0
     0.0
     0.130762955720799
     0.0
     0.869237044279201
    ```
"""
function compute_norm_projection(x::Vector, τ, p::pNorm{P}) where {P} 
    y = similar(x)
    compute_norm_projection!(y,x,τ,p)
    return y
end

"""
compute_norm_projection!(y::Vector, x::Vector, τ, p::pNorm{P}) -> y::Vector     

    Computes the projection of the vector ``x`` onto level set of the ``p``-norm specified by ``P```, of size ``τ```, i.e., solves the problem 

    ```math
        \\min_{y} & 0.5*\\|y - x\\|_2^2 \\\\
        \\text{s.t.} & \\; \\|y\\|_P \\le \\tau
    ```
    and stores the projected vector in the pre-allocated output y.

    ``P``` can be one of ``0``, ``1``, ``2``, or Inf. 

    # Example 
    ```jldoctest
    julia> x = randn(5);
    5-element Array{Float64,1}:
     0.9270834427394729
     0.5417380391296652
     1.264305123889455
     0.021179212004242474
     2.002779212447857

    julia> y = similar(x); compute_norm_projection!(y, x, 1, pNorm{1}()); y
    5-element Array{Float64,1}:
     0.0
     0.0
     0.130762955720799
     0.0
     0.869237044279201
    ```
"""
compute_norm_projection!(y::Vector, x::Vector, τ, p::pNorm{P}) where {P} = error("Not yet implemented for $(P)-norm")

function compute_norm_projection!(y::Vector, x::Vector, τ::Integer, p::pNorm{0})
    y .= x
    if norm(x,0) ≤ τ
        return
    end
    # TODO: Find the τ-th biggest element of abs(y) using order statistics
    yτ = sort(abs.(y),rev=true)[τ]
    y[abs.(y) .< yτ] .= 0.0
    return 
end

function compute_norm_projection!(y::Vector, x::Vector, τ::Real, p::pNorm{1})
    c = norm(x,1)
    if c ≤ τ
        y .= x
        return 
    end
    sign_x = sign.(x)
    abs_x = abs.(x)
    sort!(abs_x,rev=true)
    cumsum_abs_x = cumsum(abs_x)
    n = length(x)    
    found = findlast(x->x>0,[abs_x[j] - (cumsum_abs_x[j]-τ)/j for j=1:n])
    θ = (cumsum_abs_x[found] - τ)/found
    y .= sign_x .* max.(abs.(x) .- θ, 0)
    return
end

function compute_norm_projection!(y::Vector, x::Vector, τ::Real, p::pNorm{2})
    c = norm(x,2)
    y .= x
    if c ≤ τ
        return 
    end
    y .= x .* (τ/c)
    return 
end

function compute_norm_projection!(y::Vector, x::Vector, τ::Real, p::pNorm{Inf})
    c = norm(x,Inf)
    y .= x 
    if c ≤ τ
        return 
    end 
    y[y .> c] = c 
    y[y .< -c] = -c
end

# The pq Norm of a matrix X applies the p-norm to each row of X, followed by the q-norm applied to each column of x
# See, e.g., https://en.wikipedia.org/wiki/Matrix_norm 

compute_row_norms(x) = reshape(sum(x.^2, dims=2).^(0.5),size(x,2))
compute_column_norms(x) = reshape(sum(x.^2, dims=1).^(0.5),size(x,1))

struct pqNorm{P,Q} <: MatrixNorm 
    function pqNorm{P,Q}() where {P,Q}
        validate_p_value(P)
        validate_p_value(Q)
        new()
    end
end

dual_norm(pq::pqNorm{P,Q}) where {P,Q} = pNorm{1/(1-1/P),1/(1-1/Q)}()
dual_norm(pq::pqNorm{0,Q}) where {Q} = error("The (0, $(Q))-norm is not a true norm and does not have a corresponding dual")
dual_norm(pq::pqNorm{P,0}) where {P} = error("The (0, $(P))-norm is not a true norm and does not have a corresponding dual")
dual_norm(pq::pqNorm{1,1}) = pqNorm{Inf,Inf}()
dual_norm(pq::pqNorm{1,2}) = pqNorm{Inf,2}()
dual_norm(pq::pqNorm{1,Inf}) = pqNorm{Inf,1}()
dual_norm(pq::pqNorm{2,1}) = pqNorm{2,Inf}()
dual_norm(pq::pqNorm{2,2}) = pqNorm{2,2}()
dual_norm(pq::pqNorm{2,Inf}) = pqNorm{2,1}()
dual_norm(pq::pqNorm{Inf,1}) = pqNorm{1,Inf}()
dual_norm(pq::pqNorm{Inf,2}) = pqNorm{1,2}()
dual_norm(pq::pqNorm{Inf,Inf}) = pqNorm{1,1}()

function compute_norm(x::Matrix, pq::pqNorm{P,Q}) where {P,Q} 
    return sum(sum(abs.(x).^(P),dims=1).^(Q/P))^(1/Q)
end

compute_norm(x::Matrix, pq::pqNorm{0,Q}) where {Q} = norm( squeeze(sum(abs.(x).^Q,dims=2).^(1/Q)), 0)
compute_norm(x::Matrix, pq::pqNorm{P,0}) where {P} = norm( squeeze(sum(abs.(x).^P,dims=1).^(1/P)), 0)
compute_norm(x::Matrix, pq::pqNorm{0,Inf}) = norm( squeeze(maximum(abs.(x),dims=2)), 0)
compute_norm(x::Matrix, pq::pqNorm{Inf,0}) = norm( squeeze(maximum(abs.(x),dims=1)), 0)


compute_norm_projection!(y::Matrix, x::Matrix, τ, pq::pqNorm{P,Q}) where {P,Q} = error("Not yet implemented for ($(P),$(Q))-norm")

function compute_norm_projection!(y::Matrix, x::Matrix, τ, pq::pqNorm{0,2}) 
    col_norms = compute_column_norms(x)
    y .= x
    if norm(col_norms,0) ≤ τ 
        return
    end
    projected_col_norms = compute_norm_projection(col_norms, τ, pNorm{0}())
    y[:, findall(map(iszero, projected_col_norms))] .= 0.0
    return
end

function compute_norm_projection!(y::Matrix, x::Matrix, τ, pq::pqNorm{2,0}) 
    row_norms = compute_row_norms(x)
    y .= x
    if norm(row_norms,0) ≤ τ 
        return
    end
    projected_row_norms = compute_norm_projection(row_norms, τ, pNorm{0}())
    y[findall(map(iszero, projected_row_norms)),:] .= 0.0
    return
end

function compute_norm_projection!(y::Matrix, x::Matrix, τ, pq::pqNorm{1,2}) 
    col_norms = compute_column_norms(x)
    y .= x
    if norm(col_norms, 1) ≤ τ 
        return
    end
    projected_col_norms = compute_norm_projection(col_norms, τ, pNorm{1}())
    for i = 1:size(y,2)
        if col_norms[i] < 1e-6
            continue
        end
        y[:,i] .= y[:,i] .* projected_col_norms[i]/col_norms[i]
    end
    return
end

function compute_norm_projection!(y::Matrix, x::Matrix, τ, pq::pqNorm{2,1}) 
    row_norms = compute_row_norms(x)
    y .= x
    if norm(row_norms, 1) ≤ τ 
        return
    end
    projected_row_norms = compute_norm_projection(row_norms, τ, pNorm{1}())
    for i = 1:size(y,2)
        if row_norms[i] < 1e-6
            continue
        end
        y[i,:] .= y[i,:] .* projected_row_norms[i]/row_norms[i]
    end
    return
end


# The p-Schatten Norm of a matrix X is simply the vector p-norm on the vector of singular values of X

struct SchattenNorm{P} <: MatrixNorm 
    function SchattenNorm{P}() where {P}
        validate_p_value(P)
        new()
    end
end

dual_norm(p::SchattenNorm{P}) where {P} = SchattenNorm{1/(1-1/P)}()
dual_norm(p::SchattenNorm{2}) = SchattenNorm{2}()
dual_norm(p::SchattenNorm{1}) = SchattenNorm{Inf}()
dual_norm(p::SchattenNorm{Inf}) = SchattenNorm{1}()
dual_norm(p::SchattenNorm{0}) = error("The 0-norm is not a true norm and does not have a corresponding dual")

function compute_norm(x::Matrix, p::SchattenNorm{P}) where {P}
    F = svd(x)
    return compute_norm(F.S, pNorm{P}())
end

compute_norm(x::Matrix, p::SchattenNorm{2}) = norm(vec(x),2)

function compute_norm_projection!(y::Matrix, x::Matrix, τ, p::SchattenNorm{P}) where {P}
    F = svd(x)
    s = compute_norm_projection(F.S, τ, pNorm{P}())
    y .= (F.U * Diagonal(s) * F.Vt)
    return 
end


end