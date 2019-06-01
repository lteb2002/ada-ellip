
module SVDD
using Distributed
using LinearAlgebra
using Optim, LineSearches


#求解SVDD
function solveSVDD(dataset::Matrix,initX::Vector,punish::Number=100.0)
  xx = dataset
  dataSize = size(xx,1)
  featureNum = size(xx,2)
  #总变量长度
  varLen = 1 + featureNum + dataSize
  println("Length of variables:",varLen)

  C = punish#松驰变量惩罚系数
  beta=1#障碍函数惩罚系数
  ifFirst=true

  #基于某数据点计算约束函数的值

  function computeHi(i::Number,xi::Array,a::Array,R::Number,eb::Array)
    return  (xi-a)' * (xi-a) - R^2 - eb[i] ^2
  end

  #计算a的梯度
  function gra(x::Array,a::Array,R::Number,eb::Array)
    g = @distributed (+) for i in 1:dataSize
    xi= xx[i,:]
    hi=computeHi(i,xi,a,R,eb)
    g= if hi > 0
      #-4 * beta * (xi-a) * hi
      - 2 * beta * (xi-a)
    else
      zeros(length(a))
    end
  end
  return g
end

#计算R的梯度
function grR(x::Array,a::Array,R::Number,eb::Array)
  g = @distributed (+) for i in 1:dataSize
  xi= xx[i,:]
  hi=computeHi(i,xi,a,R,eb)
  g= if hi > 0
    #4 *beta * hi * R
    2 * R* beta
  else
    0
  end
end
return 2R  - g
end

#计算松驰变量的梯度
function grEb(x::Array,a::Array,R::Number,eb::Array)
  function feb(i,xi,a,R,eb)
    hi=computeHi(i,xi,a,R,eb)
    gi= if hi > 0
      #4 * beta *  hi * eb[i]
      2 * eb[i]* beta
    else
      0
    end
    return 2 * C * eb[i] - gi
  end
  greb =[feb(i,x[i,:],a,R,eb) for i in 1:dataSize]
  return Vector(greb)
end

#障碍函数 barrier function
function bf(x::Array,a::Array,R::Number,eb::Array)
  g = @distributed (+) for i in 1:dataSize
  xi= xx[i,:]
  hi=computeHi(i,xi,a,R,eb)
  g= if hi > 0
    #hi^2
    hi
  else
    0
  end
end
return g
end

#拆解总向量为各个分变量
function disassemble(y::Array)
  R=y[1]
  a=y[2 : 1+featureNum]
  eb=y[featureNum+2:end]
  return (R,a,eb)
end

#目标函数
function obj(y::Array)
  dis=disassemble(y)
  R=dis[1]
  a=dis[2]
  eb=dis[3]
  obj1 = R^2  + C * sum(eb .^ 2) + beta * bf(xx, a, R, eb)
  return obj1
end

#总梯度
function g!(G,y::Array)
  dis=disassemble(y)
  R=dis[1]
  a=dis[2]
  eb=dis[3]
  gg = []
  push!(gg,R)
  graa = gra(xx,a,R,eb)
  grEbb = grEb(xx,a,R,eb)
  gravs = vcat(gg,graa,grEbb)
  #println(size(gravs))
  #println(gravs)
  G .= gravs
end


  #变量长度：1为R，数据维度为a，数据量为eb
  x0=vcat([100],initX,zeros(dataSize).+1)
  maxStep=100
  error=1.0
  currentStep = 0
  stepSize = 1
  tol=1.0E-6
  error2=1.0
  while (error>tol) && currentStep < maxStep
    currentStep += 1
    res=optimize(obj, g!,x0, BFGS(),
                             Optim.Options(
                             iterations = 1000))
    x0 = Optim.minimizer(res)
    (R,a,eb)=disassemble(x0)
    #println("result = ", x0 )
    newError=bf(xx,a,R,eb)
    error2=norm(disassemble(x0)[3])
    stepSize= abs(error-newError)
    #println("step size:",stepSize)
    error = newError
    beta *= 3
    #println("Step:",currentStep,",Objective value: ",Optim.minimum(res),", error:",error)
  end
  #println("result = ", x0 )
  #println("error:",error)
  (rr,aa,eb)=disassemble(x0)
  println("Iterations:",currentStep)
  # if error>0.01
  #   rr =sqrt(rr^2+sum(eb.^2)/dataSize+sqrt(error/dataSize))
  # end
  return (abs(rr),aa,eb,error)
end



end
