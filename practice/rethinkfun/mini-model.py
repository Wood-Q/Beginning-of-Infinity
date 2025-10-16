X=[[10,3],[20,3],[25,3],[28,2.5],[30,2],[35,2.5],[40,2.5]]
y=[60,85,100,120,140,145,163]

w=[0.0,0.0,0.0]
lr=0.0001
num_iterations=10000

for i in range(num_iterations):
    #建立y的预测值数组y_pred
    y_pred=[w[0]+w[1]*x[0]+w[2]*x[1] for x in X]
    #损失函数MSE
    loss=sum([(y_pred[i]-y[i])**2 for i in range(len(y))])/len(y)
    #计算w0的梯度
    #用预测值减去真实值，再除以样本数量
    dw0=sum([(y_pred[i]-y[i]) for i in range(len(y))])/len(y)
    dw1=sum([(y_pred[i]-y[i])*X[i][0] for i in range(len(y))])/len(y)
    dw2=sum([(y_pred[i]-y[i])*X[i][1] for i in range(len(y))])/len(y)
    #进行梯度下降
    w[0]=w[0]-lr*dw0
    w[1]=w[1]-lr*dw1
    w[2]=w[2]-lr*dw2

    if i%100==0:
        print(f"Iteration {i+1}, Loss: {loss}")

print(f"Final weights: {w}")
print(f"Final loss: {loss}")