# CompositeConvex.jl

A level-set method for solving composite convex problems, i.e., problems of the form 
```
min_{x} h(c(x))
```
where `h(z)` is a closed, convex function (with easy to project-onto level sets) and `c(x)` is a smooth mapping, often linear. 

The full details of the algorithm are found in Ch. 4 of my PhD thesis, [available here](https://open.library.ubc.ca/collections/ubctheses/24/items/1.0355402). 


If you find this work useful, please cite 

```
C. Da Silva, "Large-scale optimization algorithms for missing data completion and inverse problems", University of British Columbia, PhD Thesis. 2017.
```

In BibTex, this is 
```
@phdthesis{da2017large,
  title={Large-scale optimization algorithms for missing data completion and inverse problems},
  author={Da Silva, Curt},
  year={2017},
  school={University of British Columbia}
}
```