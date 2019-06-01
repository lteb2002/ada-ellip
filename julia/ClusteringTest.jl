
#本程序对的结果进行评估
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
path="E:\\papers\\svdd\\datasets\\"
#数据文件名数组
#ds=["iris","abalone","magic","segment","shuttle"]
ds=["shuttle"]

for f in ds
    fn=path*f*".csv"
    println("Loading data from $fn--------------------------")
    csv = CSV.read(fn)
    X = convert(Array{Float64,2}, csv[:,1:end-1])

    cls=AdaSVDD.adaCompetitiveLearn(2,X)
    (XX,y) = ART.extractXandY(cls)
    # wq=AdaSVDD.calculateWQ(cls)
    # println("wq:$wq")
    k=length(cls)
    println("k detected by ada-svdd:$k")
    clms = Dict()
    clms[:ada_svdd]=y
    clms[:kmeans]=doKMeans(XX,k)
    #clms[:dbscan]=doDBscan(XX)
    #clms[:ward]=doHierarchical(XX)
    #clms[:affinity]=doAffinity(XX)
    #clms[:optics]=doOptics(XX)
    for ke in keys(clms)
        cls=clms[ke]
        cNum=length(unique(cls))
        println("$ke---cluster num.:$cNum------")
        q= evaluateWQ(XX,cls)
        println("q:$q")
        #dunn = evaluateDunn(X,cls)
        try
            si = evaluateSilhouette(XX,cls)
            println("si:$si")
            dbi=evaluateDBI(XX,cls)
            println("dbi:$dbi")
            # cal=evaluateCalinski(XX,cls)
            # println("cal:$cal")
            #println("dunn:$dunn")
        catch ex
        end
    end
end
