module MinComposite

using LinearAlgebra
using OptimPackNextGen
using Printf

import ..Utils: affine_model!

export ProblemParams, SubproblemOptParams, value_function!, min_composite


struct ProblemParams
    c::Function
    ∇c_adjoint::Function 
    gauge_projection::Function
    dual_gauge::Union{Function, Nothing}
end

mutable struct IntermediateVars{T}
    r::Vector{T}
    z::Vector{T}
    tau::Real
end

struct SubproblemOptParams{T}
    mem::Integer
    lower::Union{T,Vector{T},Nothing}
    upper::Union{T,Vector{T},Nothing}
    ftol::Tuple{Real,Real}
    maxiter::Integer
    maxeval::Integer
    function SubproblemOptParams{T}(mem::Integer=10,
                                    lower::Union{T,Vector{T},Nothing}=nothing,
                                    upper::Union{T,Vector{T},Nothing}=nothing,
                                    ftol::Tuple{Real,Real}=(0.0,1e-3),
                                    maxiter::Integer=20,
                                    maxeval::Integer=100) where {T}
        mem > 0 || error("mem must be > 0, got $(mem)")
        ftol[1] ≥ 0.0 || error("ftol[1] must be >= 0, got $(ftol[1])")
        ftol[2] ≥ 0.0 || error("ftol[2] must be >= 0, got $(ftol[2])")
        maxiter > 0 || error("maxiter must be > 0, got $(maxiter)")
        maxeval > 0 || error("maxeval must be > 0, got $(maxeval)")
        new(mem,lower,upper,ftol,maxiter,maxeval)
    end
end

function value_function_objective!(x, df, params::ProblemParams, info::IntermediateVars{T})::Float64 where {T}
    params.c(info.r, x)
    params.gauge_projection(info.z, info.r, info.tau)
    info.r .= info.r .- info.z
    params.∇c_adjoint(df, x, info.r)
    return 0.5*norm(info.r)^2
end

function value_function!(x::Vector{T}, params::ProblemParams, info::IntermediateVars, opt_params::SubproblemOptParams{T}) where {T}
    obj! = (x,g)->value_function_objective!(x,g,params,info)
    x .= vmlmb(obj!, x, mem=opt_params.mem, ftol=opt_params.ftol, maxiter=opt_params.maxiter, maxeval=opt_params.maxeval)

    f = obj!(x,zeros(size(x)))
    if params.dual_gauge != nothing
        df = -params.dual_gauge(info.r)::T
    else
        df = convert(0.0, T)
    end
    return f, df
end

"""
min_composite(A, b, gauge_projection, x0, dims, opt_params, dual_gauge)

Value function approach to solve problems of the form 
    min_{x} h(A*x - b)

for a non-smooth, often convex, function h(z) with easy to project on level sets. 
"""
function min_composite(A::Matrix{T}, 
                       b::Vector{T},
                       gauge_projection::Function,                                                                  
                       opt_params::SubproblemOptParams{T},
                       dual_gauge::Union{Function, Nothing}=nothing;
                       num_outer_iter::Integer = 20,
                       x0::Union{Vector{T},Nothing}=nothing,
                       initial_tau::Real=0.0,
                       use_newton::Bool=true,
                       verbose::Bool=false) where {T}
    c = (y,x)->affine_model!(y,A,x,b)
    ∇c_adjoint = (y,x,r)->mul!(y,A',r)
    return min_composite(c, ∇c_adjoint, gauge_projection, size(A), opt_params, dual_gauge, 
                        num_outer_iter=num_outer_iter, x0=x0, initial_tau=initial_tau, use_newton=use_newton, verbose=verbose)
end

"""
min_composite(c, ∇c_adjoint, gauge_projection, x0, dims, opt_params, dual_gauge)

Value function approach to solve problems of the form 
    min_{x} h(c(x))

"""
function min_composite(c::Function, 
                       ∇c_adjoint::Function, 
                       gauge_projection::Function,                        
                       dims::Tuple{Integer,Integer}, 
                       opt_params::SubproblemOptParams{T},    
                       dual_gauge::Union{Function, Nothing}=nothing;                                        
                       num_outer_iter::Integer=20,
                       x0::Union{Vector{T}, Nothing}=nothing,
                       initial_tau::Real=0.0,
                       use_newton::Bool=true,
                       verbose::Bool=false) where {T}
    if dual_gauge == nothing
        use_newton = false
    end
    problem_params = ProblemParams(c,∇c_adjoint,gauge_projection,dual_gauge)
    m,n = dims
    if x0 == nothing
        x = zeros(dims[2])
    else
        x = copy(x0)
    end
    τ1 = initial_tau
    info = IntermediateVars{T}(zeros(m),zeros(m),initial_tau)
    vf! = function(x,τ)
        info.tau = τ
        f,df = value_function!(x,problem_params, info, opt_params)
        return f, df
    end

    for i in 1:num_outer_iter
        if use_newton
            f1, df1 = vf!(x,τ1)
            verbose && @printf("itr %3d - τ: %3.3e f: %3.3e df: %3.3e\n",i,τ1,f1,df1)
            τ1 = τ1 - f1/df1            
        else
            if i==1
                f1, df1 = vf!(x,τ1)
                τ2 = max((1.001)*τ1,τ1 + 0.001)
            end
            verbose && @printf("itr %3d - τ: %3.3e f: %3.3e \n",i,τ1,f1)
            f2, df2 = vf!(x,τ2)                
            τ_new = τ2 - f2*(τ2 - τ1)/(f2 - f1)
            τ1, f1, df1 = τ2, f2, df2 
            τ2 = τ_new
        end
    end
    return x
end

end