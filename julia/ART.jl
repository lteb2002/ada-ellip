

module ART
using Distributed
using LinearAlgebra
using Optim, LineSearches

export competitiveLearn,ARTCluster

#根据簇数量及数据进行竞争型学习
function competitiveLearn(clNum::Int,dataSet::Matrix)
    iter=100
    learnRate=0.01
    #初始化所有簇中心
    (rows,cols)=size(dataSet)
    cls=[ARTCluster(dataSet[i,:],Array{Float64}(undef, 0, cols),[],0,0) for i in 1:clNum]
    for it in 1:iter
        for r in 1:rows
            x=dataSet[r,:]
            winner=checkWinner(cls,x)
            winner.centroid = winner.centroid + learnRate * (x - winner.centroid)
        end
    end
    cls=loadDataForCluster(cls,dataSet)
    return cls
end

#为每个簇装载数据
function loadDataForCluster(cls::Array,dataSet::Matrix)
    (rows,cols)=size(dataSet)
    #初始化每个簇的数据为空
    for cl in cls
        cl.dataSet=Array{Float64}(undef, 0, cols)
    end
    for r in 1:rows
        x=dataSet[r,:]
        winner=checkWinner(cls,x)
        winner.dataSet=vcat(winner.dataSet,x')
    end
    return cls
end

#根据x点找出最近的簇中心
function checkWinner(clusters::Array,x::Vector)
    map=Dict([(clu,norm(clu.centroid-x)) for clu in clusters])
    minDis=min(values(map)...)
    for (k,v) in map
        if(v==minDis)
            return k
        end
    end
end

function extractXandY(cls::Array)
    xx = vcat([c.dataSet for c in cls]...)
    y = vcat([ones(size(c.dataSet,1))*k for (k,c) in enumerate(cls)]...)
    return (xx,y)
end


#ART簇
mutable struct ARTCluster
    centroid#簇中心
    dataSet#簇中的数据
    svdd
    svddR
    svddError
end


end
