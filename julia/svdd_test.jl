

push!(LOAD_PATH, "./")
push!(LOAD_PATH, "./svdd")

using DataFrames
using CSV

input="E:\\papers\\svdd\\datasets\\iris.csv"
output="E:\\papers\\svdd\\datasets\\iris2.csv"

dataset = convert(Array{Float64,2}, CSV.read(input)[:,1:end-1])
labels = convert(Array{Any,1}, CSV.read(input)[:,end])

using TSne, Statistics, MLDatasets
dataset = tsne(dataset, 2, 50, 1000, 20.0);

function labelsAsInts(labels)
    set = Set(labels)
    maps=Dict(zip([x for x in set],1:length(set)))
    #println(typeof(maps))
    ls=[maps[label] for label in labels]
    return ls
end

ints= labelsAsInts(labels)

using Plots
theplot = scatter(dataset[:,1], dataset[:,2], legend=false, marker=(2,2,:auto,stroke(0)), color=Int.(ints[1:size(dataset,1)]))

using LinearAlgebra
using SVDD
(RR,aa,eb,error)=SVDD.solveSVDD(dataset,dataset[1,:],1)
println("R: ",RR)
println("a: ",aa)
println("norm of EB:",norm(eb))

scatter!([aa[1]],[aa[2]])

using LinearAlgebra
ls=[]
for i in 1:size(dataset,1)
    xi=dataset[i,:]
    dis=sqrt((xi-aa)'*(xi-aa))
    println(RR-dis)
    if(dis>RR)
        push!(ls,xi)
    end
end
println(length(ls))





function genBorders(num,RR,aa)
    border1=rand(num,length(aa))
    for i in 1:length(aa)-1
        border1[:,i]=((aa[i]-RR):(2*RR/num):(aa[i]+RR))[1:num]
    end
    fs(row)=sqrt(sum((row[1:(length(row)-1)]-aa[1:(length(row)-1)]).^2))
    for i in 1:size(border1,1)
        border1[i,size(dataset,2)]= RR^2 + fs(border1[i,:])
    end
    border2=copy(border1)
    border2[:,size(border1,2)]=-border1[:,size(border1,2)]
    border = vcat(border1,border2)
    border = hcat(border,["border" for x in 1:size(border,1)])
    cen= hcat(aa',["center"])
    border = vcat(border,cen)
    return border
end


borders=genBorders(50,RR,aa)
dataset = hcat(dataset,labels)
dataset = vcat(dataset,borders)

using Tables
header1=[Symbol("x$i") for i in 1:(size(dataset,2)-1)]
push!(header1,:label)
CSV.write(output,Tables.table(dataset,header=header1))



function labelsAsInts(labels)
    set = Set(labels)
    maps=Dict(zip([x for x in set],1:length(set)))
    #println(typeof(maps))
    ls=[maps[label] for label in labels]
    return ls
end

ints= labelsAsInts(labels)


using TSne, Statistics, MLDatasets
dataset = tsne(dataset, 2, 50, 1000, 20.0);


using Plots
theplot = scatter(Y[:,1], Y[:,2], marker=(2,2,:auto,stroke(0)), color=Int.(ints[1:size(Y,1)]))
