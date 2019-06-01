
push!(LOAD_PATH, "./")
push!(LOAD_PATH, "./svdd")
using RDatasets: dataset
using DataFrames
using CSV
using AdaSVDD
using ART

#using ClusterEvaluation
include("ClusterEvaluation.jl")
#数据保存位置
fn="E:\\papers\\svdd\\datasets\\shuttle.csv"
csv = CSV.read(fn)
X = convert(Array{Float64,2}, csv[:,1:end-1])

cls=AdaSVDD.adaCompetitiveLearn(2,X)
(eigs,weights)=AdaSVDD.explainEigen(cls)

for i in 1:size(eigs,1)
    println(eigs[i,:])
end

for i in 1:size(weights,1)
    println(weights[i,:])
end
