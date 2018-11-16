module CompositeConvex

export pNorm, pqNorm, SchattenNorm, compute_norm, dual_norm, compute_norm_projection

include("Norms.jl")
import .Norms: pNorm, pqNorm, SchattenNorm, compute_norm, dual_norm, compute_norm_projection, compute_norm_projection!

end 
