using Test

m,n = 1000,100
num_noise = 10
A = randn(m,n)
x = randn(n)
b = A*x
e = zeros(m)
e[rand(1:m,num_noise)]  = 1000*randn(num_noise)
b_noisy = b + e

