module CompositeConvex

export pNorm, 
       pqNorm, 
       SchattenNorm, 
       compute_norm, 
       dual_norm, 
       compute_norm_projection!,
       affine_model!,
       ProblemParams,
       SubproblemOptParams,
       min_composite

include("Norms.jl")
import .Norms: pNorm, pqNorm, SchattenNorm, compute_norm, dual_norm, compute_norm_projection!

include("Utils.jl")
import .Utils: affine_model!

include("MinComposite.jl")
import .MinComposite: ProblemParams, SubproblemOptParams, min_composite

end 
