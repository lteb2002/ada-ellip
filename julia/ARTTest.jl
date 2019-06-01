
push!(LOAD_PATH, "./")
push!(LOAD_PATH, "./svdd")

using DataFrames
using CSV

input="/home/reremouse/julia_work/test/data/iris.csv"
output="/home/reremouse/julia_work/test/data/iris2.csv"

dataset = convert(Array{Float64,2}, CSV.read(input)[:,1:end-1])
labels = convert(Array{Any,1}, CSV.read(input)[:,end])

function labelsAsInts(labels)
    set = Set(labels)
    maps=Dict(zip([x for x in set],1:length(set)))
    #println(typeof(maps))
    ls=[maps[label] for label in labels]
    return ls
end

ints= labelsAsInts(labels)

using TSne, Statistics, MLDatasets
Y = tsne(dataset, 2, 50, 1000, 20.0);

using LinearAlgebra
ff=svd(Y)
println(ff.S[1]/sum(ff.S))


using Plots
gr()

theplot = scatter(Y[:,1], Y[:,2], marker=(2,2,:auto,stroke(0)),legend=false, color=Int.(ints[1:size(Y,1)]))

using ART
cls = ART.competitiveLearn(3,Y)

using AdaSVDD
cls=AdaSVDD.adaCompetitiveLearn(2,Y)
wq=AdaSVDD.calculateWQ(cls)
y = vcat([ones(size(c.dataSet,1))*k for (k,c) in enumerate(cls)]...)
(xx,y)=ART.extractXandY(cls)

cts=zeros(length(cls),length(cls[1].centroid))
for (i,cl) in enumerate(cls)
    cts[i,:] = cl.centroid
    w=AdaSVDD.calculateFirstWeight(cl)
    println(w)
end

scatter!(cts[:,1], cts[:,2],legend=false)



using SVDD
cts2=zeros(length(cls),length(cls[1].centroid))
for (i,cl) in enumerate(cls)
    (RR,aa,eb,error)=SVDD.solveSVDD(cl.dataSet,cl.centroid,1000)
    # #println(error)
    # if(error>0.01)
    #     cts2[i,:] = cl.centroid
    # else
        cts2[i,:] = aa'
        cl.svddR=RR
        cl.svdd=aa
        cl.svddError=error
    # end
end
scatter!(cts2[:,1], cts2[:,2],legend=false,markercolor=:red,markershape=:+,markersize=7)



for cl in cls
    if cl.svddR != 0
        println("R:",cl.svddR)
        for i in 1:size(cl.dataSet,1)
            p=cl.dataSet[i,:]
            dis=norm(p-cl.svdd)
            println(dis)
            if dis > cl.svddR
                println(dis,"",p)
            end
        end
    end
end

println(cts2)

(RR,aa,eb,error)=SVDD.solveSVDD(cls[3].dataSet,cls[3].centroid,10000)

inspectdr()
theplot = scatter(Y[:,1], Y[:,2], marker=(3,3,:auto,stroke(0)),legend=false, color=Int.(ints[1:size(Y,1)]))

function drawCircle(cenX::Number,cenY::Number,r::Number)
    x = (cenX-r):0.01:(cenX+r)
    #println(-(x .- cenX).^2 .+r^2)
    y1 = cenY .+ sqrt.(abs.(-(x .- cenX).^2 .+r^2 ))
    y2= cenY .- sqrt.(abs.(-(x .- cenX).^2 .+r^2 ))
    dd = [x y1;x y2]
    plot!(dd[:,1],dd[:,2],seriestype=:scatter,markercolor=:red,markerstrokecolor=:red,markerstrokestyle=:none,markershape=:circle,markersize=0.01)
end
ii=1
drawCircle(cls[ii].svdd[1],cls[ii].svdd[2],cls[ii].svddR)
