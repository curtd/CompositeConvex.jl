module MinComposite
using LinearAlgebra
using OptimPackNextGen


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
    x = vmlmb(obj!, x, mem=opt_params.mem, ftol=opt_params.ftol, maxiter=opt_params.maxiter, maxeval=opt_params.max_eval)

    f = obj!(x,zeros(size(x)))
    if params.dual_gauge != nothing
        df = -params.dual_gauge(info.r)::T
    else
        df = convert(0.0, T)
    end
    return f, df
end

function min_composite(A::Matrix{T}, 
                       b::Vector{T},
                       gauge_projection::Function,                        
                       x0::Vector{T}, 
                       dual_gauge::Union{Function, nothing}=nothing, 
                       opt_params::SubproblemOptParams{T}) where {T}
    c = (y,x)->affine_model!(y,A,x,b)
    ∇c_adjoint = (y,x,r)->mul!(y,A',r)
    return min_composite(c, ∇c_adjoint, gauge_projection, x0, size(A), dual_gauge, opt_params)
end

"""
Value function approach to solve problems of the form 
    min_{x} h(c(x))

"""
function min_composite(c::Function, 
                       ∇c_adjoint::Function, 
                       gauge_projection::Function, 
                       x0::Vector{T}, 
                       dims::Tuple{Integer,Integer},
                       dual_gauge::Union{Function, nothing}=nothing, 
                       opt_params::SubproblemOptParams{T}) where {T}
    problem_params = ProblemParams(c,∇c_adjoint,gauge_projection,dual_gauge)
    
end

end