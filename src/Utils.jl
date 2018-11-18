module Utils
using LinearAlgebra

"""
affine_model!(y,A,x,b)

Simple affine model, ``y = Ax - b```, result stored in ``y```
"""
function affine_model!(y,A,x,b)
    mul!(y,A,x)
    y .= y .- b
    return nothing
end

end