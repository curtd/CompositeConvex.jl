module CompositeConvex

export pNorm, 
       pqNorm, 
       SchattenNorm, 
       compute_norm, 
       dual_norm, 
       compute_norm_projection,
       affine_model!

include("Norms.jl")
import .Norms: pNorm, pqNorm, SchattenNorm, compute_norm, dual_norm, compute_norm_projection, compute_norm_projection!

include("MinComposite.jl")

include("Utils.jl")
import .Utils: affine_model!

end 
