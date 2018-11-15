module MinComposite
using LinearAlgebra

struct ProblemParams
    c::Function
    ∇c_adjoint::Function 
    gauge_projection::Function
    dual_gauge::Union{Function, Nothing}
end

mutable struct IntermediateVars{Tin, Tout}
    r::Vector{Tin}
    z::Vector{Tout}
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

function value_function_objective!(x, df, params::ProblemParams, info::IntermediateVars{Tin,Tout})::Float64 where {Tin,Tout}
    params.c(info.r, x)
    params.gauge_projection(info.z, info.r, info.tau)
    info.r .= info.r .- info.z
    params.∇c_adjoint(df, info.r)
    return 0.5*norm(info.r)^2
end

function value_function!(x, params::ProblemParams, info::IntermediateVars{Tin,Tout}, opt_params::SubproblemOptParams(Tin)) where {Tin, Tout}
    obj! = (x,g)->value_function_objective!(x,g,params,info)
    x = vmlmb(obj!, x, mem=opt_params.mem, ftol=opt_params.ftol, maxiter=opt_params.maxiter, maxeval=opt_params.max_eval)

    f = obj!(x,zeros(size(x)))
    if params.dual_gauge != nothing
        df = -params.dual_gauge(info.r)::Tin
    else
        df = convert(0.0, Tin)
    end
    return f, df
end

function min_composite(A::Matrix{T}, 
                       gauge_projection::Function,                        
                       x0::Vector{T}, 
                       dual_gauge::Union{Function, nothing}=nothing, 
                       opt_params::SubproblemOptParams{T}) where {T}
    c = (y,x)->mul!(y,A,x)
    ∇c_adjoint = (y,x)->mul!(y,A',x)
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