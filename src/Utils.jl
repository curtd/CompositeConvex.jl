module Utils

function affine_model!(y,A,x,b)
    mul!(y,A,x)
    y .= y .- b
    return nothing
end

end