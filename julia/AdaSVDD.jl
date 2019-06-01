

module AdaSVDD
using Distributed
using LinearAlgebra
using Optim, LineSearches
using ART

leastWeight = 0.7

#include("ART.jl")
#using ART

#计算首个奇异值所占的比例
function calculateFirstWeight(cluster::ARTCluster)
    return calculateFirstWeight(cluster.dataSet)
end

#
function calculateFirstWeight(dataSet::Matrix)
    F=svd(dataSet)
    svs=F.S
    return svs[1]/sum(svs)
end

#计算所有簇的加权Q值
function calculateWQ(cls::Array)
    total = sum([length(c.dataSet) for c in cls])
    wq= sum([calculateFirstWeight(c.dataSet)*length(c.dataSet)/total for c in cls])
    return wq
end

#计算所有簇的加权Q值
function explainEigen(cls::Array)
    dim=size(cls[1].dataSet,2)
    cNum=length(cls)
    eigs=zeros(cNum,dim)
    weights=zeros(cNum,dim)
    for (k,cl) in enumerate(cls)
        ds = cl.dataSet
        F=svd(ds')
        svs=sum(F.S)
        eigs[k,:]=[e/svs for e in F.S]
        weight=F.U[:,1]
        println(norm(weight))
        weights[k,:]=weight
    end
    return (eigs,weights)
end



#自适应地进行竞争型学习
function adaCompetitiveLearn(leastClsNum::Int,dataSet::Matrix,clusters=[])
    cls=competitiveLearn(leastClsNum,dataSet)
    for cl in cls
        w=calculateFirstWeight(cl)
        if w <  leastWeight
            adaCompetitiveLearn(2,cl.dataSet,clusters)
        else
            push!(clusters,cl)
        end
    end
    return clusters
end


end
